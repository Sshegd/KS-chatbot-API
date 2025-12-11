# main.py — KS Chatbot Backend (FastAPI + Groq HTTP + Firebase)
# Uses Groq via raw HTTPS calls to https://api.groq.com/openai/v1/chat/completions
# No 'groq' Python package required.
import os
import json
import requests
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re
import difflib
from collections import defaultdict
import uuid
import logging

# -----------------------------
# Logging & env
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ks-backend")

load_dotenv()

# Core env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # model on Groq
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

if not FIREBASE_DATABASE_URL:
    raise Exception("FIREBASE_DATABASE_URL missing in environment")

# Globals
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]
credentials = None
app = FastAPI(title="KS Chatbot Backend (Groq HTTP)", version="4.0")

# Ensure directories
os.makedirs("tts_audio", exist_ok=True)
os.makedirs("data", exist_ok=True)

try:
    from fastapi.staticfiles import StaticFiles
    app.mount("/tts", StaticFiles(directory="tts_audio"), name="tts")
except Exception:
    logger.warning("StaticFiles not available; /tts mount skipped")

# =========================================================
# Pydantic Models
# =========================================================
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

# =========================================================
# TTS offline (pyttsx3) with gTTS fallback
# =========================================================
def generate_tts_audio(text: str, lang: str):
    filename = f"tts_{uuid.uuid4()}.mp3"
    filepath = os.path.join("tts_audio", filename)

    # Try pyttsx3 offline
    try:
        import pyttsx3
        engine = pyttsx3.init()
        if lang == "kn":
            # try to pick Kannada voice if present
            voices = engine.getProperty("voices")
            kn_voice = None
            for v in voices:
                vid = getattr(v, "id", "") or getattr(v, "name", "")
                if "kn" in vid.lower() or "kann" in vid.lower():
                    kn_voice = v.id
                    break
            if kn_voice:
                engine.setProperty("voice", kn_voice)
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        return f"/tts/{filename}"
    except Exception as e:
        logger.warning("pyttsx3 TTS failed: %s", e)

    # Fallback to gTTS online
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="kn" if lang == "kn" else "en")
        tts.save(filepath)
        return f"/tts/{filename}"
    except Exception as e:
        logger.error("gTTS fallback failed: %s", e)
        return None

# =========================================================
# Firebase admin initialization + helpers
# =========================================================
def initialize_firebase_credentials():
    global credentials
    if credentials:
        return
    try:
        if not SERVICE_ACCOUNT_KEY:
            raise Exception("SERVICE_ACCOUNT_KEY not set in environment.")
        info = json.loads(SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        logger.info("Firebase service account loaded.")
    except Exception as e:
        logger.exception("Cannot load Firebase credentials: %s", e)
        raise

def get_firebase_token() -> str:
    global credentials
    if not credentials:
        initialize_firebase_credentials()
    try:
        if not credentials.token or credentials.expired:
            credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception as e:
        logger.exception("Token refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {e}")

def firebase_get(path: str):
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_DATABASE_URL}/{path}.json"
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception("Firebase GET error: %s", e)
        return None

def get_language(user_id: str) -> str:
    lang = firebase_get(f"Users/{user_id}/preferredLanguage")
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
    return {"district": farm.get("district"), "taluk": farm.get("taluk")}

# =========================================================
# Groq HTTP helper (no SDK)
# =========================================================
def groq_chat_request(system_prompt: str, user_prompt: str, max_tokens: int = 600, temperature: float = 0.3) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Call Groq OpenAI-compatible Chat Completions endpoint via HTTPS.
    Returns (text, raw_response_dict_or_none)
    """
    if not GROQ_API_KEY:
        return "AI unavailable: GROQ_API_KEY not set.", None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Typical OpenAI shape: data["choices"][0]["message"]["content"]
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                # handle different shapes
                if isinstance(first, dict):
                    # openai-like
                    msg = first.get("message") or first.get("delta") or first
                    if isinstance(msg, dict):
                        content = msg.get("content") or msg.get("text") or None
                        if content:
                            return content, data
                    # direct text
                    txt = first.get("text") or first.get("content")
                    if txt:
                        return txt, data
            # fallback to possible 'text' or 'response' fields
            if "text" in data:
                return data["text"], data
            if "response" in data:
                return data["response"], data
        return (str(data), data)
    except Exception as e:
        logger.exception("Groq HTTP request failed: %s", e)
        return (f"AI generation error: {e}", None)

# =========================================================
# Load large constants from data/ks_constants.json (optional)
# =========================================================
DATA_DIR = "data"
constants_path = os.path.join(DATA_DIR, "ks_constants.json")

# default placeholders (user should provide full dicts)
STAGE_RECOMMENDATIONS = {}
FERTILIZER_BASE = {}
PESTICIDE_DB = {}
CROP_ET_BASE = {}
BASE_YIELD_TON_PER_HA = {}
DISEASE_WEATHER_RISK = {}
SYMPTOM_DB = {}
SYMPTOM_SYNONYMS = {}
CROP_SYMPTOM_WEIGHT = {}
DISEASE_META = {}
GENERAL_AGRI_TOPICS = {}
PRICE_LIST = {
    "chilli": 50, "paddy": 20, "ragi": 18, "areca": 470,
    "banana": 12, "turmeric": 120, "cotton": 40, "sugarcane": 3
}

if os.path.exists(constants_path):
    try:
        with open(constants_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            STAGE_RECOMMENDATIONS = data.get("STAGE_RECOMMENDATIONS", STAGE_RECOMMENDATIONS)
            FERTILIZER_BASE = data.get("FERTILIZER_BASE", FERTILIZER_BASE)
            PESTICIDE_DB = data.get("PESTICIDE_DB", PESTICIDE_DB)
            CROP_ET_BASE = data.get("CROP_ET_BASE", CROP_ET_BASE)
            BASE_YIELD_TON_PER_HA = data.get("BASE_YIELD_TON_PER_HA", BASE_YIELD_TON_PER_HA)
            DISEASE_WEATHER_RISK = data.get("DISEASE_WEATHER_RISK", DISEASE_WEATHER_RISK)
            SYMPTOM_DB = data.get("SYMPTOM_DB", SYMPTOM_DB)
            SYMPTOM_SYNONYMS = data.get("SYMPTOM_SYNONYMS", SYMPTOM_SYNONYMS)
            CROP_SYMPTOM_WEIGHT = data.get("CROP_SYMPTOM_WEIGHT", CROP_SYMPTOM_WEIGHT)
            DISEASE_META = data.get("DISEASE_META", DISEASE_META)
            GENERAL_AGRI_TOPICS = data.get("GENERAL_AGRI_TOPICS", GENERAL_AGRI_TOPICS)
            PRICE_LIST = data.get("PRICE_LIST", PRICE_LIST)
            logger.info("Loaded constants from %s", constants_path)
    except Exception as e:
        logger.exception("Failed loading constants: %s", e)

# Best-effort extraction from uploaded file if present (/mnt/data/main.txt)
UPLOADED_FILE = "/mnt/data/main.txt"
if os.path.exists(UPLOADED_FILE):
    try:
        txt = open(UPLOADED_FILE, "r", encoding="utf-8").read()
        if not STAGE_RECOMMENDATIONS:
            m = re.search(r"STAGE_RECOMMENDATIONS\s*=\s*({[\s\S]*?^\})\s*$", txt, re.M)
            if m:
                STAGE_RECOMMENDATIONS = eval(m.group(1))
        if not FERTILIZER_BASE:
            m = re.search(r"FERTILIZER_BASE\s*=\s*({[\s\S]*?^\})\s*$", txt, re.M)
            if m:
                FERTILIZER_BASE = eval(m.group(1))
        if not PESTICIDE_DB:
            m = re.search(r"PESTICIDE_DB\s*=\s*({[\s\S]*?^\})\s*$", txt, re.M)
            if m:
                PESTICIDE_DB = eval(m.group(1))
        logger.info("Attempted to extract constants from uploaded file (best-effort).")
    except Exception as e:
        logger.warning("Auto-extraction from uploaded file failed: %s", e)

# =========================================================
# Domain-specific helpers (soil center, weather, market, pest/disease, timeline)
# =========================================================
def soil_testing_center(user_id: str, language: str):
    loc = get_user_location(user_id)
    if not loc:
        msg = {
            "en": "Farm location not found. Update district & taluk in farmDetails.",
            "kn": "ಫಾರಂ ಸ್ಥಳದ ಮಾಹಿತಿ ಕಂಡುಬರಲಿಲ್ಲ. farmDetails ನಲ್ಲಿ ಜಿಲ್ಲೆ ಮತ್ತು ತಾಲೂಕು ನವೀಕರಿಸಿ."
        }
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

def weather_advice(language: str):
    advice = {
        "en": "Check forecast. If rain expected, delay irrigation. Mulching helps retain moisture.",
        "kn": "ಹವಾಮಾನ ವರದಿ ನೋಡಿ. ಮಳೆ ಸಂಭವಿಸಿದರೆ ನೀರಾವರಿ ತಡೆಯಿರಿ. ಮಲ್ಚಿಂಗ್ ಮಣ್ಣು ತೇವಾವಸ್ಥೆ ಉಳಿಸುತ್ತದೆ."
    }
    return advice[language], True, ["Irrigation schedule", "Mulching"]

def market_price(query: str, language: str):
    q = query.lower()
    for crop, price in PRICE_LIST.items():
        if crop in q:
            if language == "kn":
                return f"{crop.title()} ಸರಾಸರಿ ಬೆಲೆ: ₹{price}/ಕಿ.ಗ್ರಾಂ. ಸ್ಥಳೀಯ APMC ಪರಿಶೀಲಿಸಿ.", False, ["Sell at APMC", "Quality Check"]
            return f"Approx price for {crop.title()}: ₹{price}/kg. Verify with local APMC.", False, ["Sell at APMC", "Quality Check"]
    fallback = {"en": "Please specify the crop name (e.g., 'chilli price').", "kn": "ದಯವಿಟ್ಟು ಬೆಳೆ ಹೆಸರು ನೀಡಿ (ಉದಾ: 'ಮೆಣಸಿನ ಬೆಲೆ')."}
    return fallback[language], False, ["Chilli price", "Areca price"]

def pest_disease(query: str, language: str):
    q = query.lower()
    if "curl" in q:
        en = ("Symptoms indicate leaf curl virus or sucking pests. Remove severely affected shoots and apply neem oil spray.")
        kn = ("ಎಲೆ ಕರ್ಭಟ ವೈರಸ್ ಅಥವಾ ಸ್ಯಕ್ಕಿಂಗ್ ಕೀಟಗಳ ಸೂಚನೆ. ಗಂಭೀರವಾದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ.")
        return (kn if language == "kn" else en), True, ["Neem spray", "Contact Krishi Adhikari"]
    if "yellow" in q or "yellowing" in q:
        en = "Yellowing leaves may indicate nutrient deficiency or overwatering. Check soil moisture and consider soil test."
        kn = "ಎಲೆಗಳು ಹಳದಿ ಆಗುವುದು ಪೋಷಕಾಂಶ ಕೊರತೆ ಅಥವಾ ಹೆಚ್ಚಾಗಿ ನೀರು."
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

# =========================================================
# Stage recommendation engine
# =========================================================
def stage_recommendation_engine(crop_name: str, stage: str, lang: str) -> str:
    crop = (crop_name or "").lower()
    st = (stage or "").lower()
    if crop in STAGE_RECOMMENDATIONS and st in STAGE_RECOMMENDATIONS[crop]:
        val = STAGE_RECOMMENDATIONS[crop][st]
        if isinstance(val, dict):
            return val.get(lang, val.get("en", ""))
        return val
    fallback = {
        "en": f"No specific recommendation for {crop_name} at stage '{stage}'.",
        "kn": f"{crop_name} ಹಂತ '{stage}' ಗೆ ವಿಶೇಷ ಸಲಹೆ ಲಭ್ಯವಿಲ್ಲ."
    }
    return fallback[lang]

# =========================================================
# Fertilizer calculator
# =========================================================
def fertilizer_calculator(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    area_ha = None
    if isinstance(farm, dict):
        area_ha = farm.get("areaInHectares") or farm.get("area") or farm.get("landSizeHectares")
    try:
        area_ha = float(area_ha) if area_ha is not None else 1.0
    except Exception:
        area_ha = 1.0
    crop_l = (crop or "").lower()
    stage_l = (stage or "").lower()
    if crop_l in FERTILIZER_BASE and stage_l in FERTILIZER_BASE[crop_l]:
        N_per_ha, P_per_ha, K_per_ha = FERTILIZER_BASE[crop_l][stage_l]
        N = round(N_per_ha * area_ha, 2)
        P = round(P_per_ha * area_ha, 2)
        K = round(K_per_ha * area_ha, 2)
        if lang == "kn":
            text = (f"{crop.title()} - {stage.title()} ಹಂತಕ್ಕೆ ಶಿಫಾರಸು (ಒಟ್ಟು ಪ್ರದೇಶ {area_ha} ha):\n"
                    f"N: {N} kg, P2O5: {P} kg, K2O: {K} kg.\nದಯವಿಟ್ಟು ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಆಧರಿಸಿ ಸರಿ ಮಾಡಿ.")
        else:
            text = (f"Fertilizer recommendation for {crop.title()} ({stage.title()}) for {area_ha} ha:\n"
                    f"N: {N} kg, P2O5: {P} kg, K2O: {K} kg.\nAdjust based on soil test results.")
        return text, False, ["Soil test", "Buy fertilizer"]
    else:
        fallback = {
            "en": "No fertilizer template available for this crop/stage. Provide crop and stage or run soil test.",
            "kn": "ಈ ಬೆಳೆ/ಹಂತಕ್ಕೆ ಎರೆ ಖರೀದಿಗಾಗಿ ರೂಪರೆಖೆ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿ."
        }
        return fallback[lang], False, ["Soil test"]

# =========================================================
# Pesticide recommendation
# =========================================================
def pesticide_recommendation(crop: str, pest: str, lang: str) -> Tuple[str, bool, List[str]]:
    pest_l = (pest or "").lower()
    if pest_l in PESTICIDE_DB:
        rec = PESTICIDE_DB[pest_l].get(lang if lang in ["en", "kn"] else "en")
        if rec:
            return rec, False, ["Use bio-pesticide", "Contact advisor"]
    for key in PESTICIDE_DB.keys():
        if key in pest_l:
            rec = PESTICIDE_DB[key].get(lang if lang in ["en", "kn"] else "en")
            if rec:
                return rec, False, ["Use bio-pesticide", "Contact advisor"]
    fallback = {
        "en": "Pest not recognized. Provide photo or pest name (e.g., 'aphid', 'fruit borer').",
        "kn": "ಕೀಟ ಗುರುತಿಸಲಲಾಗಲಿಲ್ಲ. ಫೋಟೋ ಅಥವಾ ಕೀಟದ ಹೆಸರು ನೀಡಿ (ಉದಾ: aphid)."
    }
    return fallback[lang], False, ["Upload photo", "Contact Krishi Adhikari"]

# =========================================================
# Weather, irrigation and yield modules (kept similar)
# =========================================================
SOIL_WATER_HOLDING = {"sandy": 0.6, "loamy": 1.0, "clay": 1.2}
CROP_ET_BASE = CROP_ET_BASE or {}

def get_mock_weather_for_district(district):
    return {"temp": 30, "humidity": 70, "wind": 8, "rain_next_24h_mm": 0}

def fetch_weather_by_location(district: str):
    if not OPENWEATHER_KEY:
        return None
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather?"
               f"q={district}&appid={OPENWEATHER_KEY}&units=metric")
        r = requests.get(url, timeout=10)
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

def weather_suggestion_engine(weather, crop_stage=None, language="en"):
    temp = weather.get("temp", 30)
    humidity = weather.get("humidity", 50)
    wind = weather.get("wind", 8)
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
        "Low temperature – avoid fertilizer today.": "ಕಡಿಮೆ ತಾಪಮಾನ – ಇಂದು ರಸಗೊಬ್ಬರ ಬಳಕೆ ಬೇಡ.",
        "Rainfall occurring – stop irrigation for 24 hours.": "ಮಳೆ ಬರುತ್ತಿದೆ – 24 ಗಂಟೆಗಳ ಕಾಲ ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ.",
        "No rain – irrigation recommended today.": "ಮಳೆಯಿಲ್ಲ – ಇಂದು ನೀರಾವರಿ ಮಾಡಿರಿ.",
        "High humidity – fungal disease chances are high.": "ಹೆಚ್ಚು ತೇವಾಂಶ – ಫಂಗಸ್ ರೋಗದ ಸಾಧ್ಯತೆ ಹೆಚ್ಚು.",
        "Low humidity – increase irrigation frequency.": "ಕಡಿಮೆ ತೇವಾಂಶ – ನೀರಾವರಿ ಪ್ರಮಾಣ ಹೆಚ್ಚಿಸಿ.",
        "High wind – avoid spraying pesticides.": "ಬಲವಾದ ಗಾಳಿ – ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ಬೇಡ.",
        "Rain during flowering – flower drop likely.": "ಹೂ ಹಂತದಲ್ಲಿ ಮಳೆ – ಹೂ ಬಿದ್ದು ಹೋಗುವ ಸಾಧ್ಯತೆ.",
        "Rain coming – postpone harvest.": "ಮಳೆ ಬರಲಿದೆ – ಕೊಯ್ತನ್ನು ಮುಂದೂಡಿ."
    }
    return [mapping.get(s, s) for s in sugs]

def weather_advisory(user_id: str, language: str):
    farm = get_user_farm_details(user_id)
    if not farm or "district" not in farm:
        msg = {"en": "Farm district missing. Update farm details.", "kn": "ಫಾರಂ ಜಿಲ್ಲೆಯ ಮಾಹಿತಿ ಇಲ್ಲ. farmDetails ನವೀಕರಿಸಿ."}
        return msg[language], [], False
    district = farm["district"]
    weather = fetch_weather_by_location(district)
    if not weather:
        return ("Unable to fetch weather data.", [], False)
    suggestions = weather_suggestion_engine(weather, None, language)
    if language == "kn":
        report = (f"{district} ಹವಾಮಾನ:\n"
                  f"ಸ್ಥಿತಿ: {weather['description']}\n"
                  f"ತಾಪಮಾನ: {weather['temp']}°C\n"
                  f"ತೇವಾಂಶ: {weather['humidity']}%\n"
                  f"ಗಾಳಿ: {weather['wind']} km/h\n"
                  f"ಮಳೆ (1h): {weather['rain']} mm\n")
    else:
        report = (f"Weather in {district}:\n"
                  f"Condition: {weather['description']}\n"
                  f"Temperature: {weather['temp']}°C\n"
                  f"Humidity: {weather['humidity']}%\n"
                  f"Wind: {weather['wind']} km/h\n"
                  f"Rain (1h): {weather['rain']} mm\n")
    return report, suggestions, True

def irrigation_schedule(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    soil = (farm.get("soilType") or "loamy").lower()
    area_ha = 1.0
    if isinstance(farm, dict):
        try:
            area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0)
        except Exception:
            area_ha = 1.0
    district = farm.get("district") or "unknown"
    weather = get_mock_weather_for_district(district)
    rain_next_24 = weather.get("rain_next_24h_mm", 0)
    crop_l = (crop or "").lower()
    base_et = CROP_ET_BASE.get(crop_l, 4)
    soil_factor = SOIL_WATER_HOLDING.get(soil, 1.0)
    stage_mult = 1.0
    if "nursery" in (stage or "").lower() or "vegetative" in (stage or "").lower():
        stage_mult = 1.2
    elif "flower" in (stage or "").lower() or "panicle" in (stage or "").lower():
        stage_mult = 1.1
    elif "harvest" in (stage or "").lower():
        stage_mult = 0.8
    required_mm = base_et * stage_mult * (1.0 / soil_factor)
    if rain_next_24 >= 10:
        suggestion = {"en": "Rain expected soon. Delay irrigation and monitor soil moisture.", "kn": "ಶೀಘ್ರದಲ್ಲೇ ಮಳೆಯ ಸಂಭವನೆ. ನೀರಾವರಿ ತಡೆಯಿರಿ ಮತ್ತು ಮಣ್ಣು ஈರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿರಿ."}
        return suggestion[lang], False, ["Soil moisture check", "Delay irrigation"]
    liters_per_ha = required_mm * 10000
    total_liters = round(liters_per_ha * area_ha, 1)
    if lang == "kn":
        text = (f"{crop.title()} ({stage}) - ಶಿಫಾರಸು: ಪ್ರತಿ ದಿನ {round(required_mm,1)} mm ನೀರಾವರಿ (ಸಮಾನವಾದ ~{total_liters} ಲೀಟರ್/ದಿನಕ್ಕೆ {area_ha} ha).")
    else:
        text = (f"Recommendation for {crop.title()} ({stage}): approx {round(required_mm,1)} mm/day irrigation (~{total_liters} liters/day for {area_ha} ha).")
    return text, False, ["Soil moisture sensor", "Irrigation logs"]

def yield_prediction(crop: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    area_ha = 1.0
    if isinstance(farm, dict):
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
        found_recent = False
        for k, v in irrigation_logs.items():
            ts = v.get("timestamp", 0)
            if now - ts < 14 * 24 * 3600:
                found_recent = True
                break
        irrigation_ok = found_recent
    pest_incidents = firebase_get(f"Users/{user_id}/pestIncidents") or {}
    pest_control_ok = not (isinstance(pest_incidents, dict) and len(pest_incidents) > 0)
    fert_factor = 1.1 if fert_ok else 0.9
    irr_factor = 1.05 if irrigation_ok else 0.9
    pest_factor = 0.95 if not pest_control_ok else 1.0
    predicted_ton_per_ha = round(base * fert_factor * irr_factor * pest_factor, 2)
    total_tonnage = round(predicted_ton_per_ha * area_ha, 2)
    if lang == "kn":
        text = (f"ಅಂದಾಜು ಉತ್ಪಾದನೆ: {predicted_ton_per_ha} ಟನ್/ha. ಒಟ್ಟು ~{total_tonnage} ಟನ್ ನಿಮ್ಮ {area_ha} ha ಪ್ರದೇಶಕ್ಕೆ.\n"
                f"(ಕಾರಕಗಳು: fertilizer_ok={fert_ok}, irrigation_ok={irrigation_ok}, pest_ok={pest_control_ok})")
    else:
        text = (f"Estimated yield: {predicted_ton_per_ha} ton/ha. Total ~{total_tonnage} ton for {area_ha} ha.\n"
                f"(Factors: fertilizer_ok={fert_ok}, irrigation_ok={irrigation_ok}, pest_ok={pest_control_ok})")
    return text, False, ["Improve irrigation", "Soil test", "Pest control"]

# =========================================================
# Symptom recognition & diagnosis
# =========================================================
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
    for phrase, canonical in sorted((SYMPTOM_SYNONYMS or {}).items(), key=lambda x: -len(x[0])):
        if phrase in text:
            found.append(canonical)
    for key in (SYMPTOM_DB or {}).keys():
        if key in text:
            found.append(key)
    tokens = _tokenize(text)
    joined = " ".join(tokens)
    for key in (SYMPTOM_DB or {}).keys():
        try:
            ratio = difflib.SequenceMatcher(None, joined, key).ratio()
            if ratio >= fuzzy_threshold:
                found.append(key)
        except Exception:
            pass
    n = len(tokens)
    for L in range(2, min(6, n+1)):
        for i in range(n - L + 1):
            gram = " ".join(tokens[i:i+L])
            for phrase, canonical in (SYMPTOM_SYNONYMS or {}).items():
                if gram == phrase:
                    found.append(canonical)
    return list(found)

def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
    scores = defaultdict(float)
    evidence = defaultdict(list)
    for sk in symptom_keys:
        mapped = (SYMPTOM_DB or {}).get(sk, [])
        for cand in mapped:
            base_weight = 1.0
            if len(sk.split()) >= 2:
                base_weight += 0.25
            scores[cand] += base_weight
            evidence[cand].append(f"symptom:{sk}")
    if crop:
        crop_l = crop.lower()
        crop_map = (CROP_SYMPTOM_WEIGHT or {}).get(crop_l, {})
        for cand, boost in crop_map.items():
            scores[cand] += boost
            evidence[cand].append(f"crop_boost:{crop_l}")
    if not scores:
        return []
    total = sum(scores.values())
    ranked = []
    for cand, sc in sorted(scores.items(), key=lambda x: -x[1]):
        confidence = round(min(0.99, sc / (total + 1e-6)), 2)
        ranked.append((cand, round(sc, 2), confidence, evidence.get(cand, [])))
    return ranked

def diagnose_advanced(user_text: str, user_crop: Optional[str] = None, lang: str = "en") -> Tuple[str, bool, list]:
    if not user_text or not user_text.strip():
        fallback = {"en": "Please describe the symptoms (leaf color, spots, pests seen, part affected).", "kn": "ದಯವಿಟ್ಟು ಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ (ಎಲೆ ಬಣ್ಣ, ಕಲೆ, ಕಂಡ ಹಾಳುಕಾರಕಗಳು, ಭಾಗ ಪ್ರಭಾವಿತವಾಗಿರುವುದು)."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Describe symptoms"]
    symptom_keys = _extract_symptom_keys(user_text, fuzzy_threshold=0.58)
    if not symptom_keys:
        clauses = re.split(r"[,.;:/\\-]", user_text)
        for clause in clauses:
            keys = _extract_symptom_keys(clause, fuzzy_threshold=0.55)
            symptom_keys.extend(keys)
    symptom_keys = list(dict.fromkeys(symptom_keys))
    if not symptom_keys:
        fallback = {"en": "Couldn't identify clear symptoms. Please provide more details or upload a photo.", "kn": "ನಿರ್ದಿಷ್ಟ ಲಕ್ಷಣಗಳು ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Contact Krishi Adhikari"]
    ranked = _score_candidates(symptom_keys, user_crop)
    if not ranked:
        fallback = {"en": "No candidate pests/diseases found for those symptoms.", "kn": "ಆ ಲಕ್ಷಣಗಳಿಗೆ ಯೋಗ್ಯವಾದ ಕೀಟ/ರೋಗಗಳು ಕಂಡುಬರಲಿಲ್ಲ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Contact Krishi Adhikari"]
    top_k = ranked[:3]
    lines = []
    header = "Likely pests/diseases (top candidates):\n" if lang == "en" else "ಸರಾಸರಿ ಅನುಮಾನಿತ ರೋಗ/ಕೀಟಗಳು (ಮೇಲವರ್ಗ):\n"
    lines.append(header)
    for cand, score, conf, ev in top_k:
        meta = (DISEASE_META or {}).get(cand, {})
        meta_note = meta.get("note", "")
        lines.append(f"- {cand.title()} (confidence: {int(conf*100)}%)")
        if meta_note:
            lines.append(f"    • {meta_note}")
        lines.append(f"    • Evidence: {', '.join(ev)}")
    suggestions = ["Upload photo", "Contact Krishi Adhikari", "View prevention steps"]
    rec_texts = []
    for cand, score, conf, ev in top_k:
        key = cand.lower()
        if key in PESTICIDE_DB:
            rec = PESTICIDE_DB[key].get("en")
            if rec:
                rec_texts.append(f"For {cand.title()}: {rec}")
    if rec_texts:
        lines.append("\nSuggested interventions:")
        for r in rec_texts:
            lines.append(f"- {r}")
        suggestions.insert(0, "Pesticide recommendations")
    lines.append("\nIdentified symptoms:")
    for s in symptom_keys:
        lines.append(f"- {s}")
    final_text = "\n".join(lines)
    return final_text, False, suggestions

# =========================================================
# Weather + crop fusion
# =========================================================
def weather_crop_fusion(user_id: str, crop: str, stage: str, lang: str):
    farm = get_user_farm_details(user_id)
    district = farm.get("district", "unknown")
    weather = fetch_weather_by_location(district)
    if not weather:
        return ("Weather data unavailable.", False, ["Retry"])
    stage_advice = stage_recommendation_engine(crop, stage, lang)
    fusion = weather_suggestion_engine(weather, crop_stage=stage, language=lang)
    if lang == "kn":
        report = (
            f"{district} ಹवಾಮಾನ:\n"
            f"ತಾಪಮಾನ: {weather['temp']}°C | ತೇವಾಂಶ: {weather['humidity']}%\n"
            f"ಹಂತ: {crop} – {stage}\n\n"
            f"ಹಂತ ಸಲಹೆ:\n{stage_advice}\n\n"
            f"ಹವಾಮಾನ ಆಧಾರಿತ ಹೆಚ್ಚುವರಿ ಸಲಹೆಗಳು:\n- " + "\n- ".join(fusion)
        )
    else:
        report = (
            f"Weather in {district}:\n"
            f"Temp: {weather['temp']}°C | Humidity: {weather['humidity']}%\n"
            f"Stage: {crop} – {stage}\n\n"
            f"Stage Recommendation:\n{stage_advice}\n\n"
            f"Weather-based Additional Advice:\n- " + "\n- ".join(fusion)
        )
    return report, False, ["Fertilizer", "Pest Check", "Irrigation"]

# =========================================================
# Groq-backed crop_advisory (detailed + descriptive)
# =========================================================
def get_prompt(lang: str) -> str:
    if lang == "kn":
        return ("ನೀವು KrishiSakhi ಹೀಗೆ ವರ್ತಿಸಬೇಕು: ಕನ್ನಡದಲ್ಲಿಯೇ ವಿವರಣಾತ್ಮಕ ಕೃಷಿ ಸಲಹೆಗಳು ನೀಡಿ. "
                "ಹಂತ-ನಿರ್ದಿಷ್ಟ ಕ್ರಮಗಳು, ಡೋಸೇಜ್ ಸಾಮಾನ್ಯ ಮೌಲ್ಯಗಳು ಒದಗಿಸಿ. ಕಾರ್ಯನಿರ್ದೇಶನ ಮತ್ತು ಕಾರಣಗಳನ್ನು ನೀಡಿರಿ.")
    else:
        return ("You are KrishiSakhi — an agricultural assistant. Provide detailed and descriptive advice: "
                "1) Actionable steps (bulleted), 2) brief explanation with reasons. Use metric units. Prefer conservative guidance and advise soil tests when unsure.")

def crop_advisory(user_id: str, query: str, lang: str, session_key: str):
    sys_prompt = get_prompt(lang)
    farm = get_user_farm_details(user_id) or {}
    farm_summary = ""
    if farm:
        farm_summary = "Farm details: " + ", ".join(f"{k}={v}" for k, v in farm.items() if k in ["district", "soilType", "areaInHectares"]) + "."
    user_prompt = f"{farm_summary}\nUser query: {query}\nPlease respond with short bulleted actionable steps followed by 1-2 lines of explanation."

    text, raw = groq_chat_request(sys_prompt, user_prompt, max_tokens=600, temperature=0.3)
    return text, False, ["Crop stage", "Pest check", "Soil test"], session_key

# =========================================================
# get_latest_crop_stage
# =========================================================
def get_latest_crop_stage(user_id: str, lang: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return ("No farm activity found." if lang == "en" else "ಫಾರಂ ಚಟುವಟಿಕೆ ಕಂಡುಬರಲಿಲ್ಲ."), False, ["Add activity"]
    latest_ts = -1
    latest_crop = None
    latest_stage = None
    for crop, entries in logs.items():
        if isinstance(entries, dict):
            for act_id, data in entries.items():
                ts = data.get("timestamp", 0)
                if ts and ts > latest_ts:
                    latest_ts = ts
                    latest_crop = data.get("cropName", crop)
                    latest_stage = data.get("stage", "Unknown")
    rec = stage_recommendation_engine(latest_crop, latest_stage, lang)
    header = (f"{latest_crop} ಬೆಳೆ ಪ್ರಸ್ತುತ ಹಂತ: {latest_stage}\n\n" if lang == "kn" else f"Current stage of {latest_crop}: {latest_stage}\n\n")
    return header + rec, False, ["Next actions", "Fertilizer advice", "Pest check"]

# =========================================================
# Router — intent detection & calls
# =========================================================
def route(query: str, user_id: str, lang: str, session_key: str):
    q = query.lower().strip()

    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center"]):
        return {"response_text": soil_testing_center(user_id, lang)[0], "voice": True, "suggestions": ["Update farm details"]}

    if any(tok in q for tok in ["timeline", "activity log", "farm activity"]):
        t, v, s = farm_timeline(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    if any(tok in q for tok in ["weather", "rain", "forecast"]) and not ("stage" in q and "weather" in q):
        report, sug, voice = weather_advisory(user_id, lang)
        return {"response_text": report, "voice": voice, "suggestions": sug}

    if any(tok in q for tok in ["price", "market", "mandi"]):
        t, v, s = market_price(query, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    if "crop stage" in q or q == "stage" or "stage" in q:
        t, v, s = get_latest_crop_stage(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    if any(tok in q for tok in ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
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
            msg = ("Please provide the crop and stage (e.g., 'fertilizer for paddy tillering')" if lang == "en" else "ದಯವಿಟ್ಟು ಬೆಳೆ ಮತ್ತು ಹಂತವನ್ನು ನೀಡಿ (ಉದಾ: 'fertilizer for paddy tillering')")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = fertilizer_calculator(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    if "pesticide" in q or "spray" in q or "aphid" in q or "fruit borer" in q:
        pest = None
        for key in (PESTICIDE_DB or {}).keys():
            if key in q:
                pest = key
                break
        if not pest:
            msg = ("Please tell me the pest name or upload a photo (e.g., 'aphid')." if lang == "en" else "ದಯವಿಟ್ಟು ಕೀಟದ ಹೆಸರು ಅಥವಾ ಫೋಟೋ ನೀಡಿ (ಉದಾ: aphid).")
            return {"response_text": msg, "voice": False, "suggestions": ["Upload photo", "aphid"]}
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
            msg = ("Provide crop & stage for irrigation advice." if lang == "en" else "ನೀರಾವರಿ ಸಲಹೆಗೆ ಬೆಳೆ ಮತ್ತು ಹಂತ ನೀಡಿ.")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = irrigation_schedule(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    if "yield" in q or "estimate" in q or "production" in q:
        crop = None
        if BASE_YIELD_TON_PER_HA:
            for c in BASE_YIELD_TON_PER_HA.keys():
                if c in q:
                    crop = c
                    break
        if not crop:
            logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
            latest_crop = None; latest_ts = -1
            if isinstance(logs, dict):
                for crop_k, entries in logs.items():
                    if isinstance(entries, dict):
                        for aid, data in entries.items():
                            ts = data.get("timestamp", 0)
                            if ts and ts > latest_ts:
                                latest_ts = ts
                                latest_crop = data.get("cropName", crop_k)
            crop = latest_crop or (list(BASE_YIELD_TON_PER_HA.keys())[0] if BASE_YIELD_TON_PER_HA else "paddy")
        t, v, s = yield_prediction(crop, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    if "weather" in q and "stage" in q:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None; latest_stage = None; latest_ts = -1
        if isinstance(logs, dict):
            for crop, entries in logs.items():
                if isinstance(entries, dict):
                    for _, data in entries.items():
                        ts = data.get("timestamp", 0)
                        if ts and ts > latest_ts:
                            latest_ts = ts
                            latest_crop = data.get("cropName", crop)
                            latest_stage = data.get("stage", "")
        if not latest_crop:
            return {"response_text": "No crop found. Add crop activity.", "voice": False, "suggestions": ["Add activity"]}
        text, v, s = weather_crop_fusion(user_id, latest_crop, latest_stage, lang)
        return {"response_text": text, "voice": v, "suggestions": s}

    # Default → Groq crop_advisory
    t, v, s, sid = crop_advisory(user_id, query, lang, session_key)
    return {"response_text": t, "voice": v, "suggestions": s, "session_id": sid}

# =========================================================
# API endpoint
# =========================================================
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
        session_id=result.get("session_id", session_key),
        response_text=result.get("response_text", "Sorry, could not process."),
        language=lang,
        suggestions=result.get("suggestions", []),
        voice=True,
        audio_url=audio_url,
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )

# =========================================================
# Startup
# =========================================================
@app.on_event("startup")
def startup():
    try:
        initialize_firebase_credentials()
    except Exception as e:
        logger.warning("Firebase init failed at startup: %s", e)
    logger.info("KS Chatbot backend (Groq HTTP) started. Ensure GROQ_API_KEY is set in env.")

# =========================================================
# If executed directly, print short README
# =========================================================
if __name__ == "__main__":
    print("""
KS Chatbot Backend (Groq HTTP) — quick start
----------------------------------------
1) Install requirements:
   pip install fastapi uvicorn requests python-dotenv pyttsx3 gTTS google-auth

   Note: 'groq' Python package is NOT required; HTTP calls are used.

2) Environment variables (Render):
   FIREBASE_DATABASE_URL = https://your-project.firebaseio.com
   SERVICE_ACCOUNT_KEY = '{"type": "...", ... }'   # whole JSON as single-line string
   GROQ_API_KEY = your_groq_api_key
   OPENWEATHER_KEY = optional (if you want weather)

3) Optionally place big dicts in data/ks_constants.json:
   { "STAGE_RECOMMENDATIONS": {...}, "FERTILIZER_BASE": {...}, ... }

4) Run locally:
   uvicorn main:app --host 0.0.0.0 --port 8000

Notes:
 - On Render, add FIREBASE_DATABASE_URL, SERVICE_ACCOUNT_KEY and GROQ_API_KEY in Environment.
 - This file uses pure HTTPS to Groq; no groq SDK needed.
""")
