# main.py — KS Chatbot Backend (FastAPI + HF Inference + Firebase + gTTS fallback)
# Single-file final version (HF Inference API as primary LLM)
import os
import json
import time
import logging
import requests
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re
import difflib
from collections import defaultdict, Counter

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mixtral-8x7B-Instruct")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # optional legacy

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ks-backend")

# -----------------------------
# Ensure tts folder exists BEFORE mounting
# -----------------------------
TTS_DIR = "tts_audio"
os.makedirs(TTS_DIR, exist_ok=True)

# -----------------------------
# Try import gTTS; if missing, disable TTS
# -----------------------------
HAVE_GTTS = True
try:
    from gtts import gTTS  # type: ignore
except Exception:
    HAVE_GTTS = False
    logger.warning("gTTS not available; TTS will be disabled (install with 'pip install gTTS').")

# -----------------------------
# Firebase / Google credentials placeholders
# -----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]
credentials = None

# If you need firebase token refresh support, you can add google.oauth2.service_account usage.
try:
    from google.oauth2 import service_account  # type: ignore
    from google.auth.transport.requests import Request as GoogleAuthRequest  # type: ignore
    FIREBASE_SUPPORT = True
except Exception:
    FIREBASE_SUPPORT = False
    logger.info("google oauth libs not available; Firebase token-based calls will be disabled unless you install google-auth.")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="KS Chatbot Backend (HF)", version="1.0")

# mount static tts dir (safe now — directory exists)
from fastapi.staticfiles import StaticFiles  # placed after os.makedirs
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")

# -----------------------------
# Pydantic models
# -----------------------------
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

# -----------------------------
# Globals
# -----------------------------
active_chats: Dict[str, Any] = {}
# Local HF conversation memory can be very lightweight (store last prompt/responses)
hf_conversations: Dict[str, List[Dict[str, str]]] = {}

# ===========================
# Utility helpers
# ===========================
def safe_lower(s: Optional[str]) -> str:
    return (s or "").lower()

def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str):
    return text.split()

# -----------------------------
# TTS generation (gTTS) — returns audio path relative to mounted /tts
# -----------------------------
def generate_tts_audio(text: str, lang: str = "en"):
    if not HAVE_GTTS:
        logger.debug("TTS requested but gTTS not installed.")
        return None
    try:
        # choose language code
        lang_code = "kn" if lang == "kn" else "en"
        import uuid
        fname = f"tts_{uuid.uuid4().hex}.mp3"
        fpath = os.path.join(TTS_DIR, fname)
        tts = gTTS(text=text, lang=lang_code)
        tts.save(fpath)
        logger.info(f"Saved TTS file: {fpath}")
        return f"/tts/{fname}"
    except Exception as e:
        logger.exception("TTS generation failed: %s", e)
        return None

# -----------------------------
# HF Inference API helper
# -----------------------------
def hf_generate_text(prompt: str, model: str = HF_MODEL, max_tokens: int = 512, temperature: float = 0.2) -> Tuple[Optional[str], Optional[str]]:
    """
    Uses HuggingFace Inference API (simple text generation).
    Returns: (generated_text or None, error_message or None)
    """
    if not HF_API_KEY:
        return None, "HF_API_KEY not configured."

    endpoint = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=40)
        if r.status_code == 200:
            j = r.json()
            # HF may return list of generated tokens objects or dict; handle common shapes
            if isinstance(j, list):
                # each element may have 'generated_text'
                generated = ""
                for item in j:
                    if isinstance(item, dict) and "generated_text" in item:
                        generated += item["generated_text"]
                    elif isinstance(item, dict) and "text" in item:
                        generated += item["text"]
                return generated.strip(), None
            elif isinstance(j, dict) and "generated_text" in j:
                return j["generated_text"].strip(), None
            else:
                # sometimes HF returns a dict with 'error'
                if isinstance(j, dict) and "error" in j:
                    return None, j.get("error")
                # else try to stringify
                return json.dumps(j)[:10000], None
        else:
            # bubble up error
            try:
                err = r.json()
            except Exception:
                err = r.text
            logger.warning("HF inference returned non-200: %s %s", r.status_code, err)
            return None, f"HF error {r.status_code}: {err}"
    except Exception as e:
        logger.exception("HF inference call failed")
        return None, str(e)

# -----------------------------
# Optional: Gemini compatibility stub (kept for backward compatibility; disabled by default)
# -----------------------------
def gemini_generate_stub(query: str, lang: str = "en"):
    # If GEMINI_API_KEY present and you installed google genai, you can wire this.
    return "AI not configured (Gemini). Using local HF fallback.", False

# ===========================
# FIREBASE Helpers (lightweight)
# ===========================
def initialize_firebase_credentials():
    global credentials
    if not SERVICE_ACCOUNT_KEY:
        logger.info("SERVICE_ACCOUNT_KEY not set; Firebase operations that need tokens will fail.")
        return
    if not FIREBASE_SUPPORT:
        logger.info("google auth libs not available; install google-auth to enable Firebase token refresh.")
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
        raise Exception("Firebase credentials unavailable.")
    try:
        if not credentials.valid:
            credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception as e:
        raise Exception(f"Token refresh failed: {e}")

def firebase_get(path: str):
    if not FIREBASE_DATABASE_URL:
        logger.debug("No FIREBASE_DATABASE_URL configured.")
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

# Convenience fetchers used by modules
def get_language(user_id: str) -> str:
    lang = None
    try:
        lang = firebase_get(f"Users/{user_id}/preferredLanguage")
    except Exception:
        lang = None
    if isinstance(lang, str) and lang.lower() == "kn":
        return "kn"
    return "en"

def get_user_farm_details(user_id: str) -> Dict[str, Any]:
    data = firebase_get(f"Users/{user_id}/farmDetails")
    return data if isinstance(data, dict) else {}

def get_user_location(user_id: str):
    farm = get_user_farm_details(user_id)
    if not farm:
        return None
    return {
        "district": farm.get("district"),
        "taluk": farm.get("taluk")
    }

# ===========================
# Domain data (stage recs, fertilizer, pesticide, etc.)
# (I include your bigger dictionaries and functions but keep them concise for readability)
# ===========================

PRICE_LIST = {
    "chilli": 50, "paddy": 20, "ragi": 18, "areca": 470,
    "banana": 12, "turmeric": 120, "cotton": 40, "sugarcane": 3,
    # extended list for common Karnataka crops (30-ish)
    "maize": 18, "jowar": 15, "groundnut": 70, "sunflower": 45,
    "sesame": 110, "tur": 28, "moong": 60, "urad": 55,
    "banana": 12, "arecanut": 470, "coconut": 18, "coffee": 220,
    "tea": 150, "pepper": 450, "betel": 120, "sapota": 30,
    "grapes": 65, "tomato": 20, "brinjal": 25, "onion": 30,
    "potato": 22, "carrot": 18, "capsicum": 30, "ginger": 110,
    "mango": 40, "banana": 12, "papaya": 10, "pomegranate": 90
}

# Stage recommendations and fertilizer templates — include minimal examples (full mapping can be extended)
STAGE_RECOMMENDATIONS = {
    "paddy": {"nursery": {"en": "Maintain 2–3 cm water level; protect seedlings from pests.", "kn": "2–3 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಸಸಿಗಳನ್ನು ಕೀಟಗಳಿಂದ ರಕ್ಷಿಸಿ."}},
    "maize": {"vegetative": {"en": "Apply nitrogen; maintain soil moisture.", "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಣ್ಣು ತೇವ ಕಾಪಾಡಿ."}},
    # ... add more crops/stages as needed
}

FERTILIZER_BASE = {
    "paddy": {"nursery": (20, 10, 10), "tillering": (60, 30, 20)},
    "maize": {"vegetative": (80, 40, 20)},
    # ... add rest
}

PESTICIDE_DB = {
    "aphid": {"en": "Spray neem oil (2%) or insecticidal soap. Use yellow sticky traps.", "kn": "ನೀಮ್ ಎಣ್ಣೆ (2%) ಅಥವಾ ಸಾಬೂನು ಸಿಂಪಡಿಸಿ. ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್ ಬಳಸಿ."},
    "whitefly": {"en": "Use yellow sticky traps, neem oil (2%).", "kn": "ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್, ನೀಮ್ ಎಣ್ಣೆ (2%) ಬಳಸಿ."},
    "fruit borer": {"en": "Use Bt and pheromone traps.", "kn": "Bt ಮತ್ತು ಫೆರೋಮೋನ್ ಟ್ರಾಪ್ ಬಳಸಿ."},
    # ... add others
}

# Crop ET base and soil water holding
CROP_ET_BASE = {"paddy": 6.0, "maize": 5.5, "tomato": 4.8, "banana": 6.5}
SOIL_WATER_HOLDING = {"sandy": 0.6, "loamy": 1.0, "clay": 1.2}

# Base yields
BASE_YIELD_TON_PER_HA = {"paddy": 4.0, "maize": 3.5, "tomato": 25.0}

# Symptom DB & diagnosis helpers (compact but functional)
SYMPTOM_DB = {
    "yellow leaves": ["nutrient deficiency", "nitrogen deficiency", "leaf curl virus", "wilt"],
    "leaf curling": ["leaf curl virus", "thrips", "aphid", "whitefly"],
    "white powder": ["powdery mildew"],
    "black spots": ["leaf spot", "early blight", "anthracnose"],
    "holes in leaves": ["caterpillar", "armyworm", "grasshopper"]
}
SYMPTOM_SYNONYMS = {
    "yellowing": "yellow leaves",
    "leaf curl": "leaf curling",
    "white powdery": "white powder"
}
CROP_SYMPTOM_WEIGHT = {"paddy": {"tungro": 2.0, "blast": 1.8}, "tomato": {"late blight": 2.0}}
DISEASE_META = {"leaf curl virus": {"type": "viral", "note": "Usually transmitted by whiteflies"}}

def _extract_symptom_keys(user_text: str, fuzzy_threshold: float = 0.6):
    text = _normalize_text(user_text)
    found = []
    for phrase, canonical in sorted(SYMPTOM_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in text:
            found.append(canonical)
    for key in SYMPTOM_DB.keys():
        if key in text:
            found.append(key)
    tokens = _tokenize(text)
    joined = " ".join(tokens)
    for key in SYMPTOM_DB.keys():
        ratio = difflib.SequenceMatcher(None, joined, key).ratio()
        if ratio >= fuzzy_threshold:
            found.append(key)
    # ngram exact match for synonyms
    n = len(tokens)
    for L in range(2, min(6, n+1)):
        for i in range(n - L + 1):
            gram = " ".join(tokens[i:i+L])
            if gram in SYMPTOM_SYNONYMS:
                found.append(SYMPTOM_SYNONYMS[gram])
    return list(dict.fromkeys(found))

def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
    scores = defaultdict(float)
    evidence = defaultdict(list)
    for sk in symptom_keys:
        mapped = SYMPTOM_DB.get(sk, [])
        for cand in mapped:
            base_weight = 1.0
            if len(sk.split()) >= 2:
                base_weight += 0.25
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
    for cand, sc in sorted(scores.items(), key=lambda x: -x[1]):
        confidence = round(min(0.99, sc / (total + 1e-6)), 2)
        ranked.append((cand, round(sc,2), confidence, evidence.get(cand, [])))
    return ranked

# ===========================
# Modules: Soil center, pest/disease quick, farm timeline
# ===========================
def soil_testing_center(user_id: str, language: str):
    loc = get_user_location(user_id)
    if not loc:
        msg = {"en": "Farm location not found. Update district & taluk in farmDetails.",
               "kn": "ಫಾರಂ ಸ್ಥಳದ ಮಾಹಿತಿ ಕಂಡುಬರಲಿಲ್ಲ. farmDetails ನಲ್ಲಿ ಜಿಲ್ಲೆ ಮತ್ತು ತಾಲೂಕು ನವೀಕರಿಸಿ."}
        return msg[language], True, ["Update farm details"]
    district, taluk = loc["district"], loc["taluk"]
    centers = firebase_get(f"SoilTestingCenters/Karnataka/{district}/{taluk}")
    if not centers:
        return ("No soil test center found for your area.", True, ["Update farm details"])
    for _, info in centers.items():
        if isinstance(info, dict):
            text = f"{info.get('name')}\n{info.get('address')}\nContact: {info.get('contact')}"
            return text, True, ["Directions", "Call center"]
    return "No center data available.", True, []

def pest_disease(query: str, language: str):
    q = query.lower()
    if "curl" in q:
        en = ("Symptoms indicate leaf curl virus or sucking pests. Remove severely affected shoots and apply neem oil spray.")
        kn = ("ಎಲೆ ಕರ್ಭಟ ವೈರಸ್ ಅಥವಾ ಸ್ಯಕ್ಕಿಂಗ್ ಕೀಟಗಳ ಸೂಚನೆ. ಗಂಭೀರವಾದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ.")
        return (kn if language == "kn" else en), True, ["Neem spray", "Contact Krishi Adhikari"]
    if "yellow" in q or "yellowing" in q:
        en = "Yellowing leaves may indicate nutrient deficiency or overwatering. Check soil moisture and consider soil test."
        kn = "ಎಲೆಗಳು ಹಳದಿ ಆಗುವುದು ಪೋಷಕಾಂಶ ಕೊರತೆ ಅಥವಾ ಹೆಚ್ಚಾಗಿ ನೀರು 때문."
        return (kn if language == "kn" else en), True, ["Soil test", "Nitrogen application"]
    fallback = {"en": "Provide more symptom details or upload a photo.", "kn": "ಲಕ್ಷಣಗಳ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."}
    return fallback[language], True, ["Upload photo"]

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

# ===========================
# Weather helpers
# ===========================
def fetch_weather_by_location(location: str):
    if not OPENWEATHER_KEY:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_KEY}&units=metric"
        r = requests.get(url, timeout=8)
        j = r.json()
        if j.get("cod") != 200:
            return None
        return {
            "temp": j["main"]["temp"],
            "humidity": j["main"]["humidity"],
            "wind": j["wind"]["speed"],
            "condition": j["weather"][0]["main"],
            "description": j["weather"][0]["description"],
            "rain": j.get("rain", {}).get("1h", 0)
        }
    except Exception:
        return None

def get_mock_weather_for_district(district):
    return {"temp": 30, "humidity": 70, "wind": 8, "rain_next_24h_mm": 0}

def weather_suggestion_engine(weather, crop_stage=None, language="en"):
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
        if "flower" in st and cond == "Rain":
            suggestions.append("Rain during flowering – flower drop likely.")
        if "harvest" in st and rain > 0:
            suggestions.append("Rain coming – postpone harvest.")
    if language == "kn":
        suggestions = translate_weather_suggestions_kn(suggestions)
    return suggestions

def translate_weather_suggestions_kn(sugs):
    mapping = {
        "High heat – give afternoon irrigation and mulch.": "ಹೆಚ್ಚು ಬಿಸಿಲು – ಮಧ್ಯಾಹ್ನ ನೀರಾವರಿ ಮಾಡಿ ಮತ್ತು ಮಲ್ಚಿಂಗ್ ಮಾಡಿ.",
        "Low temperature – avoid fertilizer today.": "ಕಡಿಮೆ ತಾಪಮಾನ – ಇಂದು ರಸಗಿಬ್ಬರ ಬಳಕೆ ಬೇಡ.",
        "Rainfall occurring – stop irrigation for 24 hours.": "ಮಳೆ ಆಗುತ್ತಿದೆ – 24 ಘಂಟೆಗಳ ಕಾಲ ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ.",
        "No rain – irrigation recommended today.": "ಮಳೆಯಿಲ್ಲ – ಇಂದು ನೀರಾವರಿ ಮಾಡಿರಿ.",
        "High humidity – fungal disease chances are high.": "ಹೆಚ್ಚು ತೇವಾಂಶ – ಫಂಗಸ್ ರೋಗದ ಸಾಧ್ಯತೆ ಹೆಚ್ಚು.",
        "Low humidity – increase irrigation frequency.": "ಕಡಿಮೆ ತೇವಾಂಶ – ನೀರಾವರಿ ಹೆಚ್ಚಿಸಿ.",
        "High wind – avoid spraying pesticides.": "ಬಲವಾದ ಗಾಳಿ – ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ.",
        "Rain during flowering – flower drop likely.": "ಹೂ ಹಂತದಲ್ಲಿ ಮಳೆ – ಹೂ ಬೀಳುವ ಸಾಧ್ಯತೆ.",
        "Rain coming – postpone harvest.": "ಮಳೆಯು ಬರುತ್ತಿದೆ – ಕೊಯ್ತನ್ನು ಮುಂದೂಡಿ."
    }
    return [mapping.get(s, s) for s in sugs]

# ===========================
# Stage recs, fertilizer calculator, irrigation schedule, yield prediction
# ===========================
def stage_recommendation_engine(crop_name: str, stage: str, lang: str) -> str:
    crop = safe_lower(crop_name)
    st = safe_lower(stage)
    if crop in STAGE_RECOMMENDATIONS and st in STAGE_RECOMMENDATIONS[crop]:
        return STAGE_RECOMMENDATIONS[crop][st][lang if lang in ["en","kn"] else "en"]
    fallback = {"en": f"No specific recommendation for {crop_name} at stage '{stage}'.", "kn": f"{crop_name} ಹಂತ '{stage}' ಗೆ ವಿಶೇಷ ಸಲಹೆ ಲಭ್ಯವಿಲ್ಲ."}
    return fallback[lang if lang in ["en","kn"] else "en"]

def fertilizer_calculator(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    area_ha = None
    if isinstance(farm, dict):
        area_ha = farm.get("areaInHectares") or farm.get("area") or farm.get("landSizeHectares")
    try:
        area_ha = float(area_ha) if area_ha is not None else 1.0
    except Exception:
        area_ha = 1.0
    crop_l = safe_lower(crop)
    stage_l = safe_lower(stage)
    if crop_l in FERTILIZER_BASE and stage_l in FERTILIZER_BASE[crop_l]:
        N_per_ha, P_per_ha, K_per_ha = FERTILIZER_BASE[crop_l][stage_l]
        N = round(N_per_ha * area_ha, 2)
        P = round(P_per_ha * area_ha, 2)
        K = round(K_per_ha * area_ha, 2)
        if lang == "kn":
            text = (f"{crop.title()} - {stage.title()} ಹಂತಕ್ಕೆ ಶಿಫಾರಸು (ಒಟ್ಟು ಪ್ರದೇಶ {area_ha} ha):\nN: {N} kg, P2O5: {P} kg, K2O: {K} kg.\nದಯವಿಟ್ಟು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಆಧರಿಸಿ ಸರಿ ಮಾಡಿ.")
        else:
            text = (f"Fertilizer recommendation for {crop.title()} ({stage.title()}) for {area_ha} ha:\nN: {N} kg, P2O5: {P} kg, K2O: {K} kg.\nAdjust based on soil test results.")
        return text, False, ["Soil test", "Buy fertilizer"]
    else:
        fallback = {"en": "No fertilizer template available for this crop/stage. Provide crop and stage or run soil test.", "kn": "ಈ ಬೆಳೆ/ಹಂತಕ್ಕೆ ಎರೆ ಖರೀದಿಗಾಗಿ ರೂಪರೆಖೆ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿ."}
        return fallback[lang if lang in ["en","kn"] else "en"], False, ["Soil test"]

def irrigation_schedule(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    soil = (farm.get("soilType") or "loamy").lower() if isinstance(farm, dict) else "loamy"
    try:
        area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0) if isinstance(farm, dict) else 1.0
    except Exception:
        area_ha = 1.0
    district = farm.get("district") if isinstance(farm, dict) else "unknown"
    weather = get_mock_weather_for_district(district)
    rain_next_24 = weather.get("rain_next_24h_mm", 0)
    crop_l = safe_lower(crop)
    base_et = CROP_ET_BASE.get(crop_l, 4)
    soil_factor = SOIL_WATER_HOLDING.get(soil, 1.0)
    stage_mult = 1.0
    stlower = safe_lower(stage)
    if "nursery" in stlower or "vegetative" in stlower:
        stage_mult = 1.2
    elif "flower" in stlower or "panicle" in stlower:
        stage_mult = 1.1
    elif "harvest" in stlower:
        stage_mult = 0.8
    required_mm = base_et * stage_mult * (1.0 / soil_factor)
    if rain_next_24 >= 10:
        suggestion = {"en": "Rain expected soon. Delay irrigation and monitor soil moisture.", "kn": "ಶೀಘ್ರದಲ್ಲೇ ಮಳೆಯ ಸಂಭವನೆ. ನೀರಾವರಿ ತಡೆಯಿರಿ ಮತ್ತು ಮಣ್ಣು ஈರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿರಿ."}
        return suggestion[lang if lang in ["en","kn"] else "en"], False, ["Soil moisture check", "Delay irrigation"]
    liters_per_ha = required_mm * 10000
    total_liters = round(liters_per_ha * area_ha, 1)
    if lang == "kn":
        text = (f"{crop.title()} ({stage}) - ಶಿಫಾರಸು: ಪ್ರತಿ день {round(required_mm,1)} mm ನೀರಾವರಿ (~{total_liters} ಲೀಟರ್/ದಿನಕ್ಕೆ {area_ha} ha).")
    else:
        text = (f"Recommendation for {crop.title()} ({stage}): approx {round(required_mm,1)} mm/day irrigation (~{total_liters} liters/day for {area_ha} ha).")
    return text, False, ["Soil moisture sensor", "Irrigation logs"]

def yield_prediction(crop: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    try:
        area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0) if isinstance(farm, dict) else 1.0
    except Exception:
        area_ha = 1.0
    crop_l = safe_lower(crop)
    base = BASE_YIELD_TON_PER_HA.get(crop_l, 2.0)
    last_fert = firebase_get(f"Users/{user_id}/lastFertilizerApplied") or {}
    fert_ok = isinstance(last_fert, dict) and last_fert.get("applied", False)
    irrigation_logs = firebase_get(f"Users/{user_id}/irrigationLogs") or {}
    irrigation_ok = False
    if isinstance(irrigation_logs, dict):
        now = datetime.utcnow().timestamp()
        for k, v in irrigation_logs.items():
            ts = v.get("timestamp", 0)
            if now - ts < 14*24*3600:
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
        text = (f"ಅಂದಾಜು ಉತ್ಪಾದನೆ: {predicted_ton_per_ha} ಟನ್/ha. ಒಟ್ಟು ~{total_tonnage} ಟನ್ ನಿಮ್ಮ {area_ha} ha ಪ್ರದೇಶಕ್ಕೆ.\n(ಕಾರಕಗಳು: fertilizer_ok={fert_ok}, irrigation_ok={irrigation_ok}, pest_ok={pest_control_ok})")
    else:
        text = (f"Estimated yield: {predicted_ton_per_ha} ton/ha. Total ~{total_tonnage} ton for {area_ha} ha.\n(Factors: fertilizer_ok={fert_ok}, irrigation_ok={irrigation_ok}, pest_ok={pest_control_ok})")
    return text, False, ["Improve irrigation", "Soil test", "Pest control"]

# ===========================
# Weather->disease prediction
# ===========================
DISEASE_WEATHER_RISK = {
    "paddy": [{"cond":"high_humidity", "disease":"blast"}, {"cond":"continuous_rain","disease":"bacterial blight"}],
    "tomato": [{"cond":"high_humidity","disease":"late blight"}],
    # ... extend
}

def classify_weather_condition(weather):
    temp = weather.get("temp",30); humidity = weather.get("humidity",60); rain = weather.get("rain",0)
    conds = []
    if humidity > 80: conds.append("high_humidity")
    if temp > 32 and humidity < 50: conds.append("high_temp_low_humidity")
    if temp > 34: conds.append("high_temp")
    if rain > 2: conds.append("rainy")
    if rain > 8:
        conds.append("continuous_rain"); conds.append("heavy_rain")
    return conds

def predict_disease_from_weather(crop, weather, lang):
    crop = safe_lower(crop)
    if crop not in DISEASE_WEATHER_RISK:
        return None
    weather_conditions = classify_weather_condition(weather)
    risks = []
    for rule in DISEASE_WEATHER_RISK[crop]:
        if rule["cond"] in weather_conditions:
            risks.append(rule["disease"])
    if not risks:
        return {"en": f"No major disease risk predicted for {crop.title()} based on current weather.", "kn": f"ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಆಧಾರದಲ್ಲಿ {crop} ಬೆಳೆಗೂ ಮುಖ್ಯ ರೋಗ ಅಪಾಯ ಕಂಡುಬರುವುದಿಲ್ಲ."}.get(lang,"No risk")
    if lang == "kn":
        text = f"{crop} ಬೆಳೆ ಹವಾಮಾನ ಆಧಾರಿತ ರೋಗ ಅಪಾಯ:\n\n" + "\n".join([f"⚠ {d} ಅಪಾಯ ಹೆಚ್ಚು" for d in risks]) + "\n\nತಡೆ ಕ್ರಮ: ನೀಮ್ ಸಿಂಪಡಣೆ / ಗಾಳಿ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ / ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
    else:
        text = f"Disease Risk Prediction for {crop.title()}:\n\n" + "\n".join([f"⚠ High risk of {d}" for d in risks]) + "\n\nPreventive actions: Neem spray / Improve aeration / Avoid waterlogging."
    return text

# ===========================
# Advanced symptom diagnosis
# ===========================
def diagnose_advanced(user_text: str, user_crop: Optional[str] = None, lang: str = "en") -> Tuple[str, bool, list]:
    if not user_text or not user_text.strip():
        fallback = {"en":"Please describe the symptoms (leaf color, spots, pests seen, part affected).", "kn":"ದಯವಿಟ್ಟು ಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ (ಎಲೆ ಬಣ್ಣ, ಕಲೆ, ಕಂಡ ಕೀಟಗಳು, ಭಾಗ ಪ್ರಭಾವಿತ)." }
        return fallback.get(lang, fallback["en"]), False, ["Upload photo","Describe symptoms"]
    symptom_keys = _extract_symptom_keys(user_text, fuzzy_threshold=0.58)
    if not symptom_keys:
        clauses = re.split(r"[,.;:/\\-]", user_text)
        for clause in clauses:
            keys = _extract_symptom_keys(clause, fuzzy_threshold=0.55)
            symptom_keys.extend(keys)
    symptom_keys = list(dict.fromkeys(symptom_keys))
    if not symptom_keys:
        fallback = {"en":"Couldn't identify clear symptoms.", "kn":"ಲಕ್ಷಣಗಳು ಗುರುತಿಸಲಾಗಲಿಲ್ಲ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo","Contact Krishi Adhikari"]
    ranked = _score_candidates(symptom_keys, user_crop)
    if not ranked:
        return {"en":"No candidate pests/diseases found for those symptoms.","kn":"ಆ ಲಕ್ಷಣಗಳಿಗೆ ಯಾವುದೇ ಕೀಟ/ರೋಗಗಳು ಸಿಗಲಿಲ್ಲ."}.get(lang,"No result"), False, ["Upload photo","Contact Krishi Adhikari"]
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
    rec_texts=[]
    for cand, score, conf, ev in top_k:
        key = cand.lower()
        if key in PESTICIDE_DB:
            rec = PESTICIDE_DB[key].get(lang if lang in ["en","kn"] else "en")
            if rec:
                rec_texts.append(f"For {cand.title()}: {rec}")
    if rec_texts:
        lines.append("\nSuggested interventions:")
        lines += [f"- {r}" for r in rec_texts]
    lines.append("\nIdentified symptoms:")
    for s in symptom_keys:
        lines.append(f"- {s}")
    final_text = "\n".join(lines)
    return final_text, False, ["Upload photo","Pesticide recommendations","Contact Krishi Adhikari"]

# ===========================
# Crop advisory: HF fallback usage
# ===========================
def get_prompt(lang: str) -> str:
    return f"You are KrishiSakhi. Respond only in {'Kannada' if lang=='kn' else 'English'} with short actionable crop advice. Be concise and farmer-friendly."

def crop_advisory(user_id: str, query: str, lang: str, session_key: str):
    # Prefer HF inference API
    try:
        # maintain simple conversation memory
        conv = hf_conversations.setdefault(session_key, [])
        conv.append({"role":"user","content":query})
        prompt = get_prompt(lang) + "\n\nUser: " + query + "\nAssistant:"
        generated, err = hf_generate_text(prompt, model=HF_MODEL, max_tokens=400, temperature=0.2)
        if err:
            logger.warning("HF generation error: %s", err)
            return f"AI error (HF): {err}", False, [], session_key
        # persist last exchange (limit memory)
        conv.append({"role":"assistant","content":generated})
        if len(conv) > 12:
            hf_conversations[session_key] = conv[-12:]
        return generated, False, ["Crop stage", "Pest check", "Soil test"], session_key
    except Exception as e:
        logger.exception("AI fallback error: %s", e)
        return f"AI error: {e}", False, [], session_key

# ===========================
# Router — intent detection + module routing
# ===========================
def route(query: str, user_id: str, lang: str, session_key: str):
    q = query.lower().strip()
    # quick keyword intents
    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center"]):
        t, v, s = soil_testing_center(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["timeline", "activity log", "farm activity"]):
        t, v, s = farm_timeline(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["weather", "forecast", "rain"]):
        report, sug, voice = weather_advisory(user_id, lang) if 'weather_advisory' in globals() else weather_advisory(user_id, lang)
        return {"response_text": report, "voice": voice, "suggestions": sug}
    if any(tok in q for tok in ["price", "market", "mandi"]):
        t, v, s = market_price(query, lang) if 'market_price' in globals() else (f"Price lookup not configured.", False, [])
        return {"response_text": t, "voice": v, "suggestions": s}
    if "crop stage" in q or q == "stage" or "stage" in q:
        t, v, s = get_latest_crop_stage(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
        # Use advanced diagnosis if symptom-like
        if any(s in q for s in ["symptom", "leaf", "spots", "yellow", "curl"]):
            diag_text, voice, sugg = diagnose_advanced(query, None, lang)
            return {"response_text": diag_text, "voice": voice, "suggestions": sugg}
        t, v, s = pest_disease(query, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if "fertilizer" in q or "fertiliser" in q:
        # try latest crop & stage
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop=None; latest_stage=None; latest_ts=-1
        if isinstance(logs, dict):
            for crop, entries in logs.items():
                if isinstance(entries, dict):
                    for aid,data in entries.items():
                        ts = data.get("timestamp",0)
                        if ts and ts>latest_ts:
                            latest_ts=ts; latest_crop=data.get("cropName", crop); latest_stage=data.get("stage","")
        if not latest_crop:
            msg = ("Please provide the crop and stage (e.g., 'fertilizer for paddy tillering')" if lang=="en" else "ದಯವಿಟ್ಟು ಬೆಳೆ ಮತ್ತು ಹಂತ ನೀಡಿ (ಉದಾ: 'fertilizer for paddy tillering')")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = fertilizer_calculator(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["pesticide", "spray", "aphid", "fruit borer"]):
        pest = None
        for key in PESTICIDE_DB.keys():
            if key in q:
                pest = key; break
        if not pest:
            msg = ("Please tell me the pest name or upload a photo (e.g., 'aphid')." if lang=="en" else "ದಯವಿಟ್ಟು ಕೀಟದ ಹೆಸರು ಅಥವಾ ಫೋಟೋ ನೀಡಿ (ಉದಾ: aphid).")
            return {"response_text": msg, "voice": False, "suggestions": ["Upload photo", "aphid"]}
        t, v, s = pesticide_recommendation("", pest, lang) if 'pesticide_recommendation' in globals() else (PESTICIDE_DB[pest].get(lang,""), False, ["Use bio-pesticide"])
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["irrigation", "water", "irrigate"]):
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop=None; latest_stage=None; latest_ts=-1
        if isinstance(logs, dict):
            for crop, entries in logs.items():
                if isinstance(entries, dict):
                    for aid,data in entries.items():
                        ts=data.get("timestamp",0)
                        if ts and ts>latest_ts:
                            latest_ts=ts; latest_crop=data.get("cropName",crop); latest_stage=data.get("stage","")
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
            latest_crop=None; latest_ts=-1
            if isinstance(logs, dict):
                for cropk, entries in logs.items():
                    if isinstance(entries, dict):
                        for aid,data in entries.items():
                            ts = data.get("timestamp",0)
                            if ts and ts>latest_ts:
                                latest_ts=ts; latest_crop=data.get("cropName", cropk)
            crop = latest_crop or list(BASE_YIELD_TON_PER_HA.keys())[0]
        t, v, s = yield_prediction(crop, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    # weather+stage fusion
    if "weather" in q and "stage" in q:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop=None; latest_stage=None; latest_ts=-1
        if isinstance(logs, dict):
            for crop, entries in logs.items():
                if isinstance(entries, dict):
                    for _,data in entries.items():
                        ts=data.get("timestamp",0)
                        if ts and ts>latest_ts:
                            latest_ts=ts; latest_crop=data.get("cropName", crop); latest_stage=data.get("stage","")
        if not latest_crop:
            return {"response_text":"No crop found. Add crop activity.", "voice": False, "suggestions":["Add activity"]}
        text, v, s = weather_crop_fusion(user_id, latest_crop, latest_stage, lang)
        return {"response_text": text, "voice": v, "suggestions": s}
    # Advanced pest/disease diagnosis by symptoms
    if any(tok in q for tok in ["symptom", "leaf", "spots", "yellowing", "curl"]):
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop=None; latest_ts=-1
        if isinstance(logs, dict):
            for crop_k, entries in logs.items():
                if isinstance(entries, dict):
                    for aid,data in entries.items():
                        ts = data.get("timestamp",0)
                        if ts and ts>latest_ts:
                            latest_ts=ts; latest_crop=data.get("cropName", crop_k)
        diag_text, voice, sugg = diagnose_advanced(query, user_crop=latest_crop, lang=lang)
        return {"response_text": diag_text, "voice": voice, "suggestions": sugg}
    # General agricultural knowledge
    gen_text, gen_voice, gen_sugg = general_agri_knowledge_engine(query, lang) if 'general_agri_knowledge_engine' in globals() else (None, None, None)
    if gen_text:
        return {"response_text": gen_text, "voice": gen_voice, "suggestions": gen_sugg}
    # Default: forward to HF-based crop advisory
    t, v, s, sid = crop_advisory(user_id, query, lang, session_key)
    return {"response_text": t, "voice": v, "suggestions": s, "session_id": sid}

# -----------------------------
# Small wrappers referenced earlier but might not be defined in this single-file snippet:
# market_price, pesticide_recommendation, get_latest_crop_stage, weather_advisory
# Provide small consistent implementations here for completeness.
# -----------------------------
def market_price(query: str, language: str):
    q = query.lower()
    for crop, price in PRICE_LIST.items():
        if crop in q:
            if language == "kn":
                return f"{crop.title()} ಸರಾಸರಿ ಬೆಲೆ: ₹{price}/ಕಿ.ಗ್ರಾಂ. ಸ್ಥಳೀಯ APMC ಪರಿಶೀಲಿಸಿ.", False, ["Sell at APMC", "Quality Check"]
            return f"Approx price for {crop.title()}: ₹{price}/kg. Verify with local APMC.", False, ["Sell at APMC", "Quality Check"]
    fallback = {"en": "Please specify the crop name (e.g., 'chilli price').", "kn": "ದಯವಿಟ್ಟು ಬೆಳೆ ಹೆಸರು ನೀಡಿ (ಉದಾ: 'ಮೆಣಸಿನ ಬೆಲೆ')."}
    return fallback[language if language in ["en","kn"] else "en"], False, ["Chilli price", "Areca price"]

def pesticide_recommendation(crop: str, pest: str, lang: str) -> Tuple[str, bool, List[str]]:
    pest_l = (pest or "").lower()
    if pest_l in PESTICIDE_DB:
        return PESTICIDE_DB[pest_l][lang if lang in ["en","kn"] else "en"], False, ["Use bio-pesticide", "Contact advisor"]
    for key in PESTICIDE_DB.keys():
        if key in pest_l:
            return PESTICIDE_DB[key][lang if lang in ["en","kn"] else "en"], False, ["Use bio-pesticide", "Contact advisor"]
    fallback = {"en": "Pest not recognized. Provide photo or pest name (e.g., 'aphid', 'fruit borer').", "kn": "ಕೀಟ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ಫೋಟೋ ಅಥವಾ ಕೀಟದ ಹೆಸರು ನೀಡಿ (ಉದಾ: aphid)."}
    return fallback[lang if lang in ["en","kn"] else "en"], False, ["Upload photo", "Contact Krishi Adhikari"]

def get_latest_crop_stage(user_id: str, lang: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return ("No farm activity found." if lang == "en" else "ಫಾರಂ ಚಟುವಟಿಕೆ ಕಂಡುಬರಲಿಲ್ಲ."), False, ["Add activity"]
    latest_ts = -1; latest_crop=None; latest_stage=None
    if isinstance(logs, dict):
        for crop, entries in logs.items():
            if isinstance(entries, dict):
                for act_id, data in entries.items():
                    ts = data.get("timestamp",0)
                    if ts and ts>latest_ts:
                        latest_ts=ts; latest_crop=data.get("cropName", crop); latest_stage=data.get("stage", "Unknown")
    rec = stage_recommendation_engine(latest_crop or "unknown", latest_stage or "unknown", lang)
    if lang == "kn":
        header = f"{latest_crop} ಬೆಳೆ ಪ್ರಸ್ತುತ ಹಂತ: {latest_stage}\n\n"
    else:
        header = f"Current stage of {latest_crop}: {latest_stage}\n\n"
    return header + rec, False, ["Next actions", "Fertilizer advice", "Pest check"]

def weather_advisory(user_id: str, language: str):
    farm = get_user_farm_details(user_id)
    if not farm or "district" not in farm:
        msg = {"en":"Farm district missing. Update farm details.", "kn":"ಫಾರಂ ಜಿಲ್ಲೆಯ ಮಾಹಿತಿ ಇಲ್ಲ. farmDetails ನವೀಕರಿಸಿ."}
        return msg[language if language in ["en","kn"] else "en"], [], False
    district = farm["district"]
    weather = fetch_weather_by_location(district) or {}
    if not weather:
        return ("Unable to fetch weather data.", [], False)
    suggestions = weather_suggestion_engine(weather, None, language)
    if language == "kn":
        report = (f"{district} ಹವಾಮಾನ:\nಸ್ಥಿತಿ: {weather.get('description','-')}\nತಾಪಮಾನ: {weather.get('temp','-')}°C\nತೇವಾಂಶ: {weather.get('humidity','-')}%\nಗಾಳಿ: {weather.get('wind','-')} km/h\nಮಳೆ (1h): {weather.get('rain',0)} mm\n")
    else:
        report = (f"Weather in {district}:\nCondition: {weather.get('description','-')}\nTemperature: {weather.get('temp','-')}°C\nHumidity: {weather.get('humidity','-')}%\nWind: {weather.get('wind','-')} km/h\nRain (1h): {weather.get('rain',0)} mm\n")
    return report, suggestions, True

# ===========================
# Endpoint
# ===========================
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    user_query = payload.user_query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    lang = get_language(payload.user_id)
    session_key = payload.session_id or f"{payload.user_id}-{lang}"
    try:
        result = route(user_query, payload.user_id, lang, session_key)
    except Exception as e:
        logger.exception("Processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    audio_url = None
    try:
        if result.get("response_text"):
            audio_url = generate_tts_audio(result["response_text"], lang)
    except Exception as e:
        logger.exception("TTS generation failed: %s", e)
    return ChatResponse(
        session_id = result.get("session_id", session_key),
        response_text = result.get("response_text", "Sorry, could not process."),
        language = lang,
        suggestions = result.get("suggestions", []),
        voice = True if HAVE_GTTS else False,
        audio_url = audio_url,
        metadata = {"timestamp": datetime.utcnow().isoformat()}
    )

# ===========================
# Startup event
# ===========================
@app.on_event("startup")
def startup():
    logger.info("Starting KS Chatbot Backend (HF inference).")
    if FIREBASE_SUPPORT:
        initialize_firebase_credentials()
    else:
        logger.info("Firebase support disabled (google-auth not installed).")
    # no need to initialize HF — we call it on demand
    # ensure TTS folder is present (already ensured earlier)
    os.makedirs(TTS_DIR, exist_ok=True)
    logger.info("TTS directory ready at %s", TTS_DIR)
