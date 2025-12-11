# main.py — KS Chatbot Backend (FastAPI + HuggingFace Router + Firebase + gTTS)
# Uses: Mixtral (mistralai/Mixtral-8x7B-Instruct-v0.1) via Hugging Face Router
# TTS: gTTS (Option A) — creates ./tts_audio/*.mp3 and mounts under /tts

import os
import json
import time
import uuid
import requests
import traceback
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from collections import defaultdict, Counter
import re
import difflib
import logging

# ---- load .env ----
load_dotenv()

# ---- Environment ----
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")  # JSON string
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face API key (router)
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")  # default Mixtral instruct
TTS_DIR = os.getenv("TTS_DIR", "tts_audio")

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ks-backend")

# ---- Ensure tts_audio directory exists ----
os.makedirs(TTS_DIR, exist_ok=True)

# ---- Attempt to import gTTS ----
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception as e:
    logger.warning("gTTS not available: %s", e)
    GTTS_AVAILABLE = False

# ---- Firebase / Google service account token management ----
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]

credentials = None

def initialize_firebase_credentials():
    global credentials
    if credentials:
        return
    if not SERVICE_ACCOUNT_KEY:
        logger.warning("SERVICE_ACCOUNT_KEY not set; Firebase functions may fail.")
        return
    try:
        info = json.loads(SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        logger.info("Firebase service account loaded.")
    except Exception as e:
        logger.exception("Cannot load Firebase credentials: %s", e)
        credentials = None

def get_firebase_token() -> str:
    global credentials
    if not credentials:
        initialize_firebase_credentials()
    if not credentials:
        raise RuntimeError("Firebase credentials not configured")
    try:
        if not credentials.valid or credentials.expired:
            credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception as e:
        logger.exception("Token refresh failed: %s", e)
        raise

def firebase_get(path: str):
    if not FIREBASE_DATABASE_URL:
        logger.warning("FIREBASE_DATABASE_URL not configured.")
        return None
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_DATABASE_URL}/{path}.json"
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception("Firebase GET error: %s", e)
        return None

# ---- small helpers ----
def get_user_farm_details(user_id: str) -> Dict[str, Any]:
    data = firebase_get(f"Users/{user_id}/farmDetails")
    return data if isinstance(data, dict) else {}

def get_user_location(user_id: str):
    farm = get_user_farm_details(user_id)
    if not farm:
        return None
    return {"district": farm.get("district"), "taluk": farm.get("taluk")}

# ---- FastAPI app ----
app = FastAPI(title="KS Chatbot Backend (HF Mixtral)", version="1.0")

# Mount static TTS dir
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")

# ---- Models ----
class ChatQuery(BaseModel):
    user_id: str
    user_query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    language: str
    suggestions: Optional[List[str]]
    voice: Optional[bool]
    audio_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]]

# -------------------------
# Agriculture Data & modules
# (Stage recs, fertilizer base, pesticide DB, etc.)
# -------------------------

PRICE_LIST = {
    # extended list — 30 common Karnataka crops (sample prices in ₹/kg or ₹/unit as placeholder)
    "chilli": 50, "paddy": 20, "ragi": 18, "areca": 470, "banana": 12,
    "turmeric": 120, "cotton": 40, "sugarcane": 3, "maize": 22, "maize yellow": 22,
    "groundnut": 120, "sunflower": 90, "sesame": 200, "sugarcane": 3,
    "tomato": 10, "brinjal": 18, "onion": 25, "potato": 12, "carrot": 20,
    "capsicum": 40, "banana": 12, "mango": 30, "coconut": 15, "arecanut": 470,
    "coffee": 300, "pepper": 250, "tur": 60, "moong": 80, "urad": 70
}

# Stage recommendation example (partial — keep as in your prior code)
STAGE_RECOMMENDATIONS = {
    "paddy": {
        "nursery": {"en": "Maintain 2–3 cm water level; protect seedlings from pests.", "kn": "2–3 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಸಸಿಗಳನ್ನು ಕೀಟಗಳಿಂದ ರಕ್ಷಿಸಿ."},
        "tillering": {"en": "Apply urea (N); maintain 3–5 cm water; manage weeds.", "kn": "ಯೂರಿಯಾ (N) ನೀಡಿ; 3–5 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಗಿಡ್ಮುಳ್ಳು ನಿಯಂತ್ರಿಸಿ."},
        "harvest": {"en": "Harvest when 80% grains turn golden yellow.", "kn": "80% ಧಾನ್ಯ ಬಂಗಾರದ ಬಣ್ಣವಾಗಿದಾಗ ಕೊಯಿರಿ."}
    },
    # add more crops/stages as needed...
}

# Fertilizer base sample (kept compact)
FERTILIZER_BASE = {
    "paddy": {"nursery": (20,10,10), "tillering": (60,30,20), "panicle initiation": (30,20,20)},
    "maize": {"vegetative": (80,40,20), "tasseling": (40,20,20)}
    # extend...
}

# Pesticide DB (short form)
PESTICIDE_DB = {
    "aphid": {"en": "Spray neem oil (2%) or insecticidal soap.", "kn": "ನೀಮ್ ಎಣ್ಣೆ (2%) ಸಿಂಪಡಿಸಿ."},
    "whitefly": {"en": "Use yellow sticky traps, neem oil (2%).", "kn": "ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್, ನೀಮ್ ಎಣ್ಣೆ (2%)."},
    "fruit borer": {"en": "Apply Bacillus thuringiensis (Bt).", "kn": "ಬ್ಯಾಸಿಲಸ್ ಥುರಿಂಜಿಯೆನ್ಸಿಸ್ (Bt)."}
    # extend...
}

# SYMPTOM DB (compact)
SYMPTOM_DB = {
    "yellow leaves": ["nutrient deficiency", "nitrogen deficiency", "leaf curl virus"],
    "leaf curling": ["leaf curl virus", "thrips", "aphid", "whitefly"],
    "white powder": ["powdery mildew"],
    "black spots": ["leaf spot", "early blight", "anthracnose"]
}
SYMPTOM_SYNONYMS = {
    "yellowing": "yellow leaves",
    "leaf curl": "leaf curling",
    "white powdery": "white powder",
    "black spots on leaf": "black spots"
}
CROP_SYMPTOM_WEIGHT = {"paddy": {"tungro": 2.0, "blast": 1.8}, "tomato": {"late blight": 2.0}}

DISEASE_META = {
    "leaf curl virus": {"type": "viral", "note": "Usually transmitted by whiteflies"},
    "aphid": {"type": "insect", "note": "Sucking insect - causes honeydew"},
    "powdery mildew": {"type": "fungal", "note": "White powder on leaf surfaces"}
}

BASE_YIELD_TON_PER_HA = {
    "paddy": 4.0, "maize": 3.5, "ragi": 1.8, "banana": 20.0, "tomato": 25.0
}

# ---- small utilities for symptom extraction, scoring ----
def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str):
    return text.split()

def _extract_symptom_keys(user_text: str, fuzzy_threshold: float = 0.6):
    text = _normalize_text(user_text)
    found = []
    # synonyms exact
    for phrase, canonical in sorted(SYMPTOM_SYNONYMS.items(), key=lambda x:-len(x[0])):
        if phrase in text:
            found.append(canonical)
    # exact canonical
    for key in SYMPTOM_DB.keys():
        if key in text:
            found.append(key)
    # fuzzy over joined
    joined = " ".join(_tokenize(text))
    for key in SYMPTOM_DB.keys():
        ratio = difflib.SequenceMatcher(None, joined, key).ratio()
        if ratio >= fuzzy_threshold:
            found.append(key)
    # n-gram synonyms
    tokens = _tokenize(text)
    n = len(tokens)
    for L in range(2, min(6, n+1)):
        for i in range(n - L + 1):
            gram = " ".join(tokens[i:i+L])
            for phrase, canonical in SYMPTOM_SYNONYMS.items():
                if gram == phrase:
                    found.append(canonical)
    return list(dict.fromkeys(found))

def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
    scores = defaultdict(float)
    evidence = defaultdict(list)
    for sk in symptom_keys:
        mapped = SYMPTOM_DB.get(sk, [])
        for cand in mapped:
            base_weight = 1.0 + (0.25 if len(sk.split()) >= 2 else 0.0)
            scores[cand] += base_weight
            evidence[cand].append(f"symptom:{sk}")
    if crop:
        crop_l = crop.lower()
        crop_map = CROP_SYMPTOM_WEIGHT.get(crop_l, {})
        for cand, boost in crop_map.items():
            scores[cand] += boost
            evidence[cand].append(f"crop_boost:{crop_l}")
    if not scores:
        return []
    total = sum(scores.values())
    ranked = []
    for cand, sc in sorted(scores.items(), key=lambda x:-x[1]):
        confidence = round(min(0.99, sc / (total + 1e-6)), 2)
        ranked.append((cand, round(sc,2), confidence, evidence.get(cand, [])))
    return ranked

# -------------------------
# TTS generation (gTTS)
# -------------------------
def generate_tts_audio(text: str, lang: str) -> Optional[str]:
    """
    Generate mp3 using gTTS. Returns URL path like /tts/tts_<uuid>.mp3 or None if failed.
    """
    if not GTTS_AVAILABLE:
        logger.warning("gTTS not available — skipping TTS generation.")
        return None
    try:
        # choose language code
        code = "kn" if lang == "kn" else "en"
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(TTS_DIR, filename)
        tts = gTTS(text=text, lang=code)
        tts.save(filepath)
        logger.info("Saved TTS file: %s", filepath)
        # return mounted url
        return f"/tts/{filename}"
    except Exception as e:
        logger.exception("TTS error: %s", e)
        return None

# -------------------------
# Hugging Face Router client for text generation (Mixtral)
# -------------------------
HF_ROUTER_URL_TEMPLATE = "https://router.huggingface.co/models/{model}"

def hf_generate_text(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (text, error_message). If HF not configured or request fails, returns (None, error_message).
    Uses Hugging Face Router endpoint.
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured (HF_API_KEY)."

    url = HF_ROUTER_URL_TEMPLATE.format(model=HF_MODEL)
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": False
        },
        # optionally set "options": {"use_cache": False} or streaming -- kept simple
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        # log the status and errors
        if r.status_code == 200:
            data = r.json()
            # router returns a list or dict depending on model; handle common shapes
            if isinstance(data, dict):
                # { "generated_text": "..." } or other shapes
                text = data.get("generated_text") or data.get("text") or str(data)
                return text, None
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"], None
                # some HF models return {"generated_text": "..."} inside list
                return str(first), None
            return str(data), None
        else:
            logger.warning("HF non-200: %s %s", r.status_code, r.text[:400])
            if r.status_code == 401:
                return None, "Hugging Face authentication error (check HF_API_KEY)."
            if r.status_code == 404:
                return None, f"HF model {HF_MODEL} not found."
            if r.status_code == 429:
                return None, "Hugging Face quota / rate limit exceeded."
            # propagate error body
            return None, f"HF error {r.status_code}: {r.text}"
    except Exception as e:
        logger.exception("HF generate exception: %s", e)
        return None, f"HF request failed: {e}"

# -------------------------
# Application modules: soil, weather, market, pest/disease, timeline, fertilizer, irrigation, yield, diagnosis
# (Implementations derived from your earlier large code)
# -------------------------

# Soil testing centers
def soil_testing_center(user_id: str, language: str):
    loc = get_user_location(user_id)
    if not loc:
        msg = {
            "en": "Farm location not found. Update district & taluk in farmDetails.",
            "kn": "ಫಾರಂ ಸ್ಥಳದ ಮಾಹಿತಿ ಕಂಡುಬರಲಿಲ್ಲ. farmDetails ನಲ್ಲಿ ಜಿಲ್ಲೆ ಮತ್ತು ತಾಲೂಕು ನವೀಕರಿಸಿ."
        }
        return msg[language], True, ["Update farm details"]
    district, taluk = loc.get("district"), loc.get("taluk")
    if not district or not taluk:
        return ("No soil test center found for your area.", True, ["Update farm details"])
    centers = firebase_get(f"SoilTestingCenters/Karnataka/{district}/{taluk}")
    if not centers:
        return ("No soil test center found for your area.", True, ["Update farm details"])
    for _, info in centers.items():
        if isinstance(info, dict):
            text = f"{info.get('name')}\n{info.get('address')}\nContact: {info.get('contact')}"
            return text, True, ["Directions", "Call center"]
    return "No center data available.", True, []

# simple pest/disease quick handler
def pest_disease(query: str, language: str):
    q = (query or "").lower()
    if "curl" in q or "curling" in q:
        en = "Symptoms indicate leaf curl virus or sucking pests. Remove severely affected shoots and apply neem oil spray."
        kn = "ಎಲೆ ಕರ್ಭಟ ವೈರಸ್ ಅಥವಾ ಸ್ಯಕ್ಕಿಂಗ್ ಕೀಟಗಳ ಸೂಚನೆ. ಗಂಭೀರವಾದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ."
        return (kn if language == "kn" else en), True, ["Neem spray", "Contact Krishi Adhikari"]
    if "yellow" in q or "yellowing" in q:
        en = "Yellowing leaves may indicate nutrient deficiency or overwatering. Check soil moisture and consider soil test."
        kn = "ಎಲೆಗಳು ಹಳದಿ ಆಗುವುದು ಪೋಷಕಾಂಶ ಕೊರತೆ ಅಥವಾ ಹೆಚ್ಚಾಗಿ ನೀರು 때문."
        return (kn if language == "kn" else en), True, ["Soil test", "Nitrogen application"]
    fallback = {"en": "Provide more symptom details or upload a photo.", "kn": "ಲಕ್ಷಣಗಳ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."}
    return fallback[language], True, ["Upload photo"]

# farm timeline (as requested)
def farm_timeline(user_id: str, language: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return ("No activity logs found." if language == "en" else "ಚಟುವಟಿಕೆ ಲಾಗ್ ಕಂಡುಬರಲಿಲ್ಲ."), False, ["Add activity"]
    summaries = []
    for crop, entries in logs.items():
        if isinstance(entries, dict):
            latest_entry = None
            latest_ts = -1
            for act_id, data in entries.items():
                ts = data.get("timestamp", 0)
                if ts and ts > latest_ts:
                    latest_ts = ts
                    latest_entry = data
            if latest_entry:
                crop_name = latest_entry.get("cropName", crop)
                act = latest_entry.get("subActivity", "")
                stage = latest_entry.get("stage", "")
                if language == "kn":
                    summaries.append(f"{crop_name}: ಇತಿಚ್ಚಿನ ಚಟುವಟಿಕೆ {act} (ಹಂತ: {stage})")
                else:
                    summaries.append(f"{crop_name}: latest activity {act} (stage: {stage})")
    if not summaries:
        return ("No recent activities found." if language == "en" else "ಯಾವುದೇ ಇತ್ತೀಚಿನ ಚಟುವಟಿಕೆಗಳು ಇಲ್ಲ."), False, ["Add activity"]
    return ("\n".join(summaries), False, ["View full timeline"])

# weather fetcher
def fetch_weather_by_location(district: str):
    if not OPENWEATHER_KEY or not district:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={district}&appid={OPENWEATHER_KEY}&units=metric"
        r = requests.get(url, timeout=8)
        data = r.json()
        if data.get("cod") != 200:
            return None
        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "rain": data.get("rain", {}).get("1h", 0)
        }
    except Exception:
        return None

def get_mock_weather_for_district(district):
    return {"temp": 30, "humidity": 70, "wind": 8, "rain": 0, "condition": "Clear", "description": "clear sky"}

def weather_suggestion_engine(weather, crop_stage=None, language="en"):
    if not weather:
        return []
    temp = weather.get("temp", 30)
    humidity = weather.get("humidity", 60)
    wind = weather.get("wind", 5)
    rain = weather.get("rain", 0)
    cond = weather.get("condition", "")

    suggestions = []
    if temp > 35:
        suggestions.append("High heat – give afternoon irrigation and mulch.")
    elif temp < 15:
        suggestions.append("Low temperature – avoid fertilizer today.")
    if rain > 3:
        suggestions.append("Rainfall occurring – stop irrigation for 24 hours.")
    else:
        suggestions.append("No rain – irrigation recommended today.")
    if humidity > 80:
        suggestions.append("High humidity – fungal disease chances are high.")
    elif humidity < 35:
        suggestions.append("Low humidity – increase irrigation frequency.")
    if wind > 20:
        suggestions.append("High wind – avoid spraying pesticides.")
    if crop_stage:
        st = crop_stage.lower()
        if "flower" in st and cond.lower().startswith("rain"):
            suggestions.append("Rain during flowering – flower drop likely.")
        if "harvest" in st and rain > 0:
            suggestions.append("Rain coming – postpone harvest.")
    return suggestions

def weather_advisory(user_id: str, language: str):
    farm = get_user_farm_details(user_id)
    if not farm or "district" not in farm:
        msg = {"en": "Farm district missing. Update farm details.", "kn": "ಫಾರಂ ಜಿಲ್ಲೆಯ ಮಾಹಿತಿ ಇಲ್ಲ. farmDetails ನವೀಕರಿಸಿ."}
        return msg[language], [], False
    district = farm.get("district")
    weather = fetch_weather_by_location(district) or get_mock_weather_for_district(district)
    suggestions = weather_suggestion_engine(weather, None, language)
    if language == "kn":
        report = (f"{district} ಹವಾಮಾನ:\nಸ್ಥಿತಿ: {weather.get('description')}\nತಾಪಮಾನ: {weather.get('temp')}°C\nತೇವಾಂಶ: {weather.get('humidity')}%\nಗಾಳಿ: {weather.get('wind')} km/h\nಮಳೆ (1h): {weather.get('rain')} mm\n")
    else:
        report = (f"Weather in {district}:\nCondition: {weather.get('description')}\nTemperature: {weather.get('temp')}°C\nHumidity: {weather.get('humidity')}%\nWind: {weather.get('wind')} km/h\nRain (1h): {weather.get('rain')} mm\n")
    return report, suggestions, True

# fertilizer calculator
def fertilizer_calculator(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str,bool,List[str]]:
    farm = get_user_farm_details(user_id)
    area_ha = 1.0
    if isinstance(farm, dict):
        try:
            area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0)
        except Exception:
            area_ha = 1.0
    crop_l = (crop or "").lower(); stage_l = (stage or "").lower()
    if crop_l in FERTILIZER_BASE and stage_l in FERTILIZER_BASE[crop_l]:
        N_per_ha, P_per_ha, K_per_ha = FERTILIZER_BASE[crop_l][stage_l]
        N = round(N_per_ha * area_ha, 2); P = round(P_per_ha * area_ha, 2); K = round(K_per_ha * area_ha, 2)
        if lang == "kn":
            text = (f"{crop.title()} - {stage.title()} ಹಂತಕ್ಕೆ ಶಿಫಾರಸು ({area_ha} ha):\nN: {N} kg, P2O5: {P} kg, K2O: {K} kg.")
        else:
            text = (f"Fertilizer recommendation for {crop.title()} ({stage.title()}) for {area_ha} ha:\nN: {N} kg, P2O5: {P} kg, K2O: {K} kg.")
        return text, False, ["Soil test", "Buy fertilizer"]
    else:
        fallback = {"en": "No fertilizer template available for this crop/stage. Provide crop and stage or run soil test.", "kn": "ಈ ಬೆಳೆ/ಹಂತಕ್ಕೆ ಎರೆ ರೂಪನಿ ಲಭ್ಯವಿಲ್ಲ."}
        return fallback[lang], False, ["Soil test"]

# irrigation schedule (simplified)
SOIL_WATER_HOLDING = {"sandy": 0.6, "loamy": 1.0, "clay": 1.2}
CROP_ET_BASE = {"paddy":6.0, "maize":5.5, "tomato":4.8, "banana":6.5}

def irrigation_schedule(crop: str, stage: str, user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    soil = (farm.get("soilType") or "loamy").lower() if isinstance(farm, dict) else "loamy"
    try:
        area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0)
    except Exception:
        area_ha = 1.0
    district = farm.get("district") if isinstance(farm, dict) else "unknown"
    weather = get_mock_weather_for_district(district)
    rain_next_24 = weather.get("rain_next_24h_mm", 0)
    crop_l = (crop or "").lower()
    base_et = CROP_ET_BASE.get(crop_l, 4.0)
    soil_factor = SOIL_WATER_HOLDING.get(soil, 1.0)
    stage_mult = 1.0
    if "nursery" in (stage or "").lower() or "vegetative" in (stage or "").lower():
        stage_mult = 1.2
    elif "flower" in (stage or "").lower():
        stage_mult = 1.1
    elif "harvest" in (stage or "").lower():
        stage_mult = 0.8
    required_mm = base_et * stage_mult * (1.0 / soil_factor)
    if rain_next_24 >= 10:
        suggestion = {"en": "Rain expected soon. Delay irrigation and monitor soil moisture.", "kn": "ಶೀಘ್ರ ಮಳೆ. ನೀರಾವರಿ ತಡೆದಿರಿ."}
        return suggestion[lang], False, ["Soil moisture check", "Delay irrigation"]
    liters_per_ha = required_mm * 10000
    total_liters = round(liters_per_ha * area_ha, 1)
    if lang == "kn":
        text = f"{crop.title()} ({stage}) - ಶಿಫಾರಸು: ಪ್ರತಿ ದಿನ ~{round(required_mm,1)} mm (~{total_liters} L/day for {area_ha} ha)."
    else:
        text = f"Recommendation for {crop.title()} ({stage}): approx {round(required_mm,1)} mm/day (~{total_liters} liters/day for {area_ha} ha)."
    return text, False, ["Soil moisture sensor", "Irrigation logs"]

# yield prediction (simple heuristic)
def yield_prediction(crop: str, user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    try:
        area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0)
    except Exception:
        area_ha = 1.0
    crop_l = (crop or "").lower()
    base = BASE_YIELD_TON_PER_HA.get(crop_l, 2.0)
    last_fert = firebase_get(f"Users/{user_id}/lastFertilizerApplied") or {}
    fert_ok = isinstance(last_fert, dict) and last_fert.get("applied", False)
    irrigation_logs = firebase_get(f"Users/{user_id}/irrigationLogs") or {}
    irrigation_ok = False
    if isinstance(irrigation_logs, dict):
        now = datetime.utcnow().timestamp()
        for k, v in irrigation_logs.items():
            ts = v.get("timestamp", 0)
            if ts and now - ts < 14*24*3600:
                irrigation_ok = True
                break
    pest_incidents = firebase_get(f"Users/{user_id}/pestIncidents") or {}
    pest_control_ok = not (isinstance(pest_incidents, dict) and len(pest_incidents) > 0)
    fert_factor = 1.1 if fert_ok else 0.9
    irr_factor = 1.05 if irrigation_ok else 0.9
    pest_factor = 1.0 if pest_control_ok else 0.95
    predicted_ton_per_ha = round(base * fert_factor * irr_factor * pest_factor, 2)
    total_tonnage = round(predicted_ton_per_ha * area_ha, 2)
    if lang == "kn":
        text = f"ಅಂದಾಜು ಉತ್ಪಾದನೆ: {predicted_ton_per_ha} ಟನ್/ha. ಒಟ್ಟು ~{total_tonnage} ಟನ್ ({area_ha} ha)."
    else:
        text = f"Estimated yield: {predicted_ton_per_ha} ton/ha. Total ~{total_tonnage} ton for {area_ha} ha."
    return text, False, ["Improve irrigation", "Soil test", "Pest control"]

# disease-from-weather prediction (simple)
DISEASE_WEATHER_RISK = {"paddy": [{"cond":"high_humidity","disease":"blast"}], "tomato":[{"cond":"high_humidity","disease":"late blight"}]}

def classify_weather_condition(weather):
    temp = weather.get("temp",30); humidity = weather.get("humidity",60); rain = weather.get("rain",0)
    conds = []
    if humidity > 80: conds.append("high_humidity")
    if temp > 34: conds.append("high_temp")
    if rain > 2: conds.append("rainy")
    if rain > 8: conds.append("continuous_rain")
    if temp > 32 and humidity < 50: conds.append("high_temp_low_humidity")
    return conds

def predict_disease_from_weather(crop, weather, lang):
    crop = (crop or "").lower()
    if crop not in DISEASE_WEATHER_RISK:
        return None
    weather_conditions = classify_weather_condition(weather)
    risks = []
    for rule in DISEASE_WEATHER_RISK[crop]:
        if rule["cond"] in weather_conditions:
            risks.append(rule["disease"])
    if not risks:
        return {"en": f"No major disease risk predicted for {crop.title()} based on current weather.", "kn": f"ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಆಧಾರದಲ್ಲಿ {crop} ರೋಗ ಅಪಾಯ ಕಡಿಮೆ."}[lang]
    if lang == "kn":
        text = f"{crop} ಬೆಳೆ ಹವಾಮಾನ ಆಧಾರಿತ ರೋಗ ಅಪಾಯ:\n" + "\n".join([f"⚠ {d} ಅಪಾಯ" for d in risks])
    else:
        text = f"Disease Risk Prediction for {crop.title()}:\n" + "\n".join([f"⚠ High risk of {d}" for d in risks])
    return text

# diagnosis advanced using symptom engine
def diagnose_advanced(user_text: str, user_crop: Optional[str] = None, lang: str = "en"):
    if not user_text or not user_text.strip():
        fallback = {"en":"Please describe the symptoms (leaf color, spots, pests seen, part affected).","kn":"ದಯವಿಟ್ಟು ಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ."}
        return fallback[lang], False, ["Upload photo","Describe symptoms"]
    symptom_keys = _extract_symptom_keys(user_text, fuzzy_threshold=0.58)
    if not symptom_keys:
        clauses = re.split(r"[,.;:/\\-]", user_text)
        for clause in clauses:
            keys = _extract_symptom_keys(clause, fuzzy_threshold=0.55)
            symptom_keys.extend(keys)
    symptom_keys = list(dict.fromkeys(symptom_keys))
    if not symptom_keys:
        fallback = {"en":"Couldn't identify clear symptoms. Please provide more details or upload a photo.","kn":"ಲಕ್ಷಣಗಳು ಸ್ಪಷ್ಟವಾಗಿ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ."}
        return fallback[lang], False, ["Upload photo","Contact Krishi Adhikari"]
    ranked = _score_candidates(symptom_keys, user_crop)
    if not ranked:
        fallback = {"en":"No candidate pests/diseases found for those symptoms.","kn":"ಲಕ್ಷಣಗಳಿಗೆ ಹೊಂದುವ ಕೀಟ/ರೋಗಗಳು ಕಂಡುಬಂದಿಲ್ಲ."}
        return fallback[lang], False, ["Upload photo","Contact Krishi Adhikari"]
    top_k = ranked[:3]
    lines = []
    header = "Likely pests/diseases (top candidates):\n" if lang=="en" else "ಸರಾಸರಿ ಅನುಮಾನಿತ ರೋಗ/ಕೀಟಗಳು:\n"
    lines.append(header)
    for cand, score, conf, ev in top_k:
        meta = DISEASE_META.get(cand, {})
        meta_note = meta.get("note","")
        lines.append(f"- {cand.title()} (confidence: {int(conf*100)}%)")
        if meta_note:
            lines.append(f"    • {meta_note}")
        lines.append(f"    • Evidence: {', '.join(ev)}")
    rec_texts = []
    for cand, score, conf, ev in top_k:
        key = cand.lower()
        if key in PESTICIDE_DB:
            rec = PESTICIDE_DB[key].get(lang if lang in ["en","kn"] else "en")
            if rec:
                rec_texts.append(f"For {cand.title()}: {rec}")
    if rec_texts:
        lines.append("\nSuggested interventions:")
        for r in rec_texts:
            lines.append(f"- {r}")
    lines.append("\nIdentified symptoms:")
    for s in symptom_keys:
        lines.append(f"- {s}")
    final_text = "\n".join(lines)
    return final_text, False, ["Pesticide recommendations", "Upload photo"]

# get latest crop stage from farmActivityLogs
def get_latest_crop_stage(user_id: str, lang: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return ("No farm activity found." if lang == "en" else "ಫಾರಂ ಚಟುವಟಿಕೆ ಕಂಡುಬರಲಿಲ್ಲ."), False, ["Add activity"]
    latest_ts = -1; latest_crop = None; latest_stage = None
    for crop, entries in logs.items():
        if isinstance(entries, dict):
            for act_id, data in entries.items():
                ts = data.get("timestamp", 0)
                if ts and ts > latest_ts:
                    latest_ts = ts
                    latest_crop = data.get("cropName", crop)
                    latest_stage = data.get("stage", "Unknown")
    rec = stage_recommendation_engine(latest_crop or "", latest_stage or "", lang)
    header = (f"{latest_crop} current stage: {latest_stage}\n\n" if lang=="en" else f"{latest_crop} ಬೆಳೆ ಪ್ರಸ್ತುತ ಹಂತ: {latest_stage}\n\n")
    return header + rec, False, ["Next actions", "Fertilizer advice", "Pest check"]

def stage_recommendation_engine(crop_name: str, stage: str, lang: str) -> str:
    crop = (crop_name or "").lower(); st = (stage or "").lower()
    if crop in STAGE_RECOMMENDATIONS and st in STAGE_RECOMMENDATIONS[crop]:
        return STAGE_RECOMMENDATIONS[crop][st][lang if lang in ["en","kn"] else "en"]
    fallback = {"en": f"No specific recommendation for {crop_name} at stage '{stage}'.", "kn": f"{crop_name} ಹಂತ '{stage}' ಗೆ ಸಲಹೆ ಲಭ್ಯವಿಲ್ಲ."}
    return fallback[lang]

# -------------------------
# Router / Intent engine
# -------------------------
def route(query: str, user_id: str, lang: str, session_key: str):
    q = (query or "").lower().strip()
    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center"]):
        t, v, s = soil_testing_center(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["timeline", "activity log", "farm activity"]):
        t, v, s = farm_timeline(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["weather", "rain", "forecast"]):
        report, sug, voice = weather_advisory(user_id, lang)
        return {"response_text": report, "voice": voice, "suggestions": sug}
    if any(tok in q for tok in ["price", "market", "mandi"]):
        t, v, s = market_price(query, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if "crop stage" in q or q == "stage" or "stage" in q:
        t, v, s = get_latest_crop_stage(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
        # fallback: advanced diagnosis
        diag_text, voice, sugg = diagnose_advanced(query, user_crop=None, lang=lang)
        return {"response_text": diag_text, "voice": voice, "suggestions": sugg}
    if "fertilizer" in q or "fertiliser" in q or "apply fertilizer" in q:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None; latest_stage = None; latest_ts = -1
        if isinstance(logs, dict):
            for crop, entries in logs.items():
                if isinstance(entries, dict):
                    for aid, data in entries.items():
                        ts = data.get("timestamp", 0)
                        if ts and ts > latest_ts:
                            latest_ts = ts
                            latest_crop = data.get("cropName", crop)
                            latest_stage = data.get("stage", "")
        if not latest_crop:
            msg = ("Please provide the crop and stage (e.g., 'fertilizer for paddy tillering')" if lang=="en" else "ದಯವಿಟ್ಟು ಬೆಳೆ ಮತ್ತು ಹಂತವನ್ನು ನೀಡಿ.")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = fertilizer_calculator(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if "pesticide" in q or "spray" in q or "aphid" in q or "fruit borer" in q:
        pest = None
        for key in PESTICIDE_DB.keys():
            if key in q:
                pest = key; break
        if not pest:
            msg = ("Please tell me the pest name or upload a photo (e.g., 'aphid')." if lang=="en" else "ದಯವಿಟ್ಟು ಕೀಟದ ಹೆಸರು ಅಥವಾ ಫೋಟೋ ನೀಡಿ.")
            return {"response_text": msg, "voice": False, "suggestions": ["Upload photo","aphid"]}
        t, v, s = pesticide_recommendation("", pest, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if "irrigation" in q or "water" in q or "irrigate" in q:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None; latest_stage = None; latest_ts = -1
        if isinstance(logs, dict):
            for crop, entries in logs.items():
                if isinstance(entries, dict):
                    for aid, data in entries.items():
                        ts = data.get("timestamp", 0)
                        if ts and ts > latest_ts:
                            latest_ts = ts
                            latest_crop = data.get("cropName", crop)
                            latest_stage = data.get("stage", "")
        if not latest_crop:
            msg = ("Provide crop & stage for irrigation advice." if lang=="en" else "ನೀರಾವರಿ ಸಲಹೆಗೆ ಬೆಳೆ ಮತ್ತು ಹಂತ ನೀಡಿ.")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = irrigation_schedule(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if "yield" in q or "estimate" in q or "production" in q:
        crop = None
        for c in BASE_YIELD_TON_PER_HA.keys():
            if c in q:
                crop = c; break
        if not crop:
            logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
            latest_crop = None; latest_ts = -1
            if isinstance(logs, dict):
                for crop_k, entries in logs.items():
                    if isinstance(entries, dict):
                        for aid, data in entries.items():
                            ts = data.get("timestamp", 0)
                            if ts and ts > latest_ts:
                                latest_ts = ts; latest_crop = data.get("cropName", crop_k)
            crop = latest_crop or list(BASE_YIELD_TON_PER_HA.keys())[0]
        t, v, s = yield_prediction(crop, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    # default -> HF Mixtral advisory (if available) else canned fallback
    hf_prompt = f"You are KrishiSakhi. Respond only in {'Kannada' if lang=='kn' else 'English'} with short actionable crop advice for the user query: {query}"
    text, err = hf_generate_text(hf_prompt, max_new_tokens=256, temperature=0.2)
    if text:
        return {"response_text": text, "voice": False, "suggestions": ["Crop stage","Pest check","Soil test"]}
    else:
        # HF missing/failure -> fallback small canned message
        fallback = "AI advisor currently unavailable. I can still help with soil test, fertilizer, irrigation, pest diagnosis. Ask one of those." if lang=="en" else "AI ಸಲಹೆ ಸದ್ಯ ಲಭ್ಯವಿಲ್ಲ. ಮಣ್ಣು ಪರೀಕ್ಷೆ, ಎರೆ, ನೀರಾವರಿ, ಕೀಟನಿರ್ಣಯಕ್ಕೆ ಕೇಳಿ."
        logger.warning("HF fallback used: %s", err)
        return {"response_text": fallback, "voice": False, "suggestions": ["Soil test","Fertilizer","Pest check"]}

# pesticide_recommendation wrapper
def pesticide_recommendation(crop: str, pest: str, lang: str):
    pest_l = (pest or "").lower()
    if pest_l in PESTICIDE_DB:
        return PESTICIDE_DB[pest_l][lang if lang in ["en","kn"] else "en"], False, ["Use bio-pesticide","Contact advisor"]
    for key in PESTICIDE_DB.keys():
        if key in pest_l:
            return PESTICIDE_DB[key][lang if lang in ["en","kn"] else "en"], False, ["Use bio-pesticide","Contact advisor"]
    fallback = {"en":"Pest not recognized. Provide photo or pest name (e.g., 'aphid').","kn":"ಕೀಟ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ಫೋಟೋ ಅಥವಾ ಹೆಸರು ನೀಡಿ."}
    return fallback[lang], False, ["Upload photo","Contact Krishi Adhikari"]

def market_price(query: str, language: str):
    q = (query or "").lower()
    for crop, price in PRICE_LIST.items():
        if crop in q:
            if language == "kn":
                return f"{crop.title()} ಸರಾಸರಿ ಬೆಲೆ: ₹{price}/ಕಿ.ಗ್ರಾಂ. ಸ್ಥಳೀಯ APMC ಪರಿಶೀಲಿಸಿ.", False, ["Sell at APMC", "Quality Check"]
            return f"Approx price for {crop.title()}: ₹{price}/kg. Verify with local APMC.", False, ["Sell at APMC", "Quality Check"]
    fallback = {"en": "Please specify the crop name (e.g., 'chilli price').", "kn": "ದಯವಿಟ್ಟು ಬೆಳೆ ಹೆಸರು ನೀಡಿ (ಉದಾ: 'ಮೆಣಸಿನ ಬೆಲೆ')."}
    return fallback[language], False, ["Chilli price", "Areca price"]

# -------------------------
# Endpoint
# -------------------------
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    user_query = (payload.user_query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    # language detection via user's preferred language field
    lang = "en"
    try:
        pref = firebase_get(f"Users/{payload.user_id}/preferredLanguage")
        if isinstance(pref, str) and pref.lower() == "kn":
            lang = "kn"
    except Exception:
        lang = "en"
    session_key = payload.session_id or f"{payload.user_id}-{lang}"
    try:
        result = route(user_query, payload.user_id, lang, session_key)
    except Exception as e:
        logger.exception("Processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    audio_url = None
    try:
        if result.get("response_text"):
            # generate TTS (gTTS) — if available
            audio_url = generate_tts_audio(result["response_text"], lang)
    except Exception as e:
        logger.exception("TTS generation failed: %s", e)
    return ChatResponse(
        session_id=result.get("session_id", session_key),
        response_text=result.get("response_text", "Sorry, could not process."),
        language=lang,
        suggestions=result.get("suggestions", []),
        voice=True,
        audio_url=audio_url,
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )

# -------------------------
# Startup event: initialize firebase credentials
# -------------------------
@app.on_event("startup")
def startup():
    logger.info("Starting KS Chatbot backend...")
    # create tts dir if needed
    os.makedirs(TTS_DIR, exist_ok=True)
    # firebase
    try:
        initialize_firebase_credentials()
    except Exception as e:
        logger.exception("Firebase init failed: %s", e)
    if not HF_API_KEY:
        logger.warning("HF_API_KEY not configured — HF generation disabled; fallback responses used.")
    else:
        logger.info("Hugging Face API key present. Model: %s", HF_MODEL)
    if not GTTS_AVAILABLE:
        logger.warning("gTTS not installed — TTS disabled. Install via `pip install gTTS` to enable.")

# -------------------------
# Simple health endpoint
# -------------------------
@app.get("/health")
def health():
    return {"status":"ok", "time": datetime.utcnow().isoformat(), "hf": bool(HF_API_KEY), "gtts": GTTS_AVAILABLE}

# End of file


