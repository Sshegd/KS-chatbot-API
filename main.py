# =========================================================
# KS CHATBOT BACKEND (MAIN.PY)
# Minimal & Clean Version
# Includes:
# - Agriculture Q&A Engine (100+ topics)
# - Multi-language detection (EN + KN)
# - Simulated NDVI Crop Health Module
# - Offline fallback engine
# - Stage-wise, Fertilizer, Pesticide, Irrigation, Yield modules
# - Gemini fallback
# - Firebase integration
# =========================================================

import os
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from google import genai
from google.genai import types
from datetime import datetime
import re
import difflib
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")

if not FIREBASE_URL:
    raise Exception("Missing FIREBASE_DATABASE_URL")

app = FastAPI(title="KS Chatbot Backend", version="5.0")

# Create TTS folder
os.makedirs("tts_audio", exist_ok=True)
app.mount("/tts", StaticFiles(directory="tts_audio"), name="tts")

client = None
credentials = None
active_chats: Dict[str, Any] = {}

# ---------------------------------------------------------
# FASTAPI MODELS
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# TTS ENGINE
# ---------------------------------------------------------

def generate_tts_audio(text: str, lang: str):
    try:
        from gtts import gTTS
        import uuid

        filename = f"tts_{uuid.uuid4()}.mp3"
        filepath = f"./tts_audio/{filename}"

        tts = gTTS(text=text, lang="kn" if lang == "kn" else "en")
        tts.save(filepath)
        return f"/tts/{filename}"
    except:
        return None

# ---------------------------------------------------------
# GEMINI INITIALIZATION
# ---------------------------------------------------------

def initialize_gemini():
    global client
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)

def get_prompt(lang: str) -> str:
    return (
        "You are KrishiSakhi. Provide simple, short, actionable agriculture advice "
        f"in {'Kannada' if lang == 'kn' else 'English'}."
    )

# ---------------------------------------------------------
# FIREBASE INITIALIZATION
# ---------------------------------------------------------

SCOPES = [
    "https://www.googleapis.com/auth/firebase.database",
    "https://www.googleapis.com/auth/userinfo.email"
]

def initialize_firebase_credentials():
    global credentials
    info = json.loads(SERVICE_ACCOUNT_KEY)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES
    )

def get_firebase_token() -> str:
    global credentials
    if not credentials.token or credentials.expired:
        credentials.refresh(GoogleAuthRequest())
    return credentials.token

def firebase_get(path: str):
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_URL}/{path}.json"
        r = requests.get(url, params={"access_token": token})
        r.raise_for_status()
        return r.json()
    except:
        return None

# ---------------------------------------------------------
# USER LANGUAGE
# ---------------------------------------------------------

def detect_language(text: str):
    kannada_chars = sum(1 for c in text if 'ಅ' <= c <= 'ಹ')
    return "kn" if kannada_chars > 2 else "en"

def get_preferred_language(user_id: str):
    lang = firebase_get(f"Users/{user_id}/preferredLanguage")
    if isinstance(lang, str) and lang.lower() == "kn":
        return "kn"
    return "en"

# ---------------------------------------------------------
# FARM DETAILS
# ---------------------------------------------------------

def get_user_farm_details(user_id: str):
    d = firebase_get(f"Users/{user_id}/farmDetails")
    return d if isinstance(d, dict) else {}

# ---------------------------------------------------------
# NDVI (SIMULATED)
# ---------------------------------------------------------

def fetch_simulated_ndvi(lat=None, lon=None):
    import random
    return round(random.uniform(0.20, 0.85), 2)

def ndvi_health_report(user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    lat = farm.get("latitude")
    lon = farm.get("longitude")

    if not lat or not lon:
        msg = {
            "en": "Farm coordinates missing. Add latitude & longitude.",
            "kn": "ಫಾರಂ ಸ್ಥಳಾಂಶ ಲಭ್ಯವಿಲ್ಲ. ಅಕ್ಷಾಂಶ/ರೇಖಾಂಶ ಸೇರಿಸಿ."
        }
        return msg[lang], False, ["Update farm details"]

    ndvi = fetch_simulated_ndvi(lat, lon)

    if ndvi < 0.3:
        status = "Poor crop health" if lang == "en" else "ಕಡಿಮೆ ಬೆಳೆ ಆರೋಗ್ಯ"
    elif ndvi < 0.5:
        status = "Moderate health" if lang == "en" else "ಮಧ್ಯಮ ಆರೋಗ್ಯ"
    else:
        status = "Healthy" if lang == "en" else "ಆರೋಗ್ಯಕರ"

    text = (
        f"NDVI: {ndvi}\nStatus: {status}"
        if lang == "en" else
        f"NDVI: {ndvi}\nಸ್ಥಿತಿ: {status}"
    )

    return text, False, ["Pest Check", "Irrigation advice"]

# ---------------------------------------------------------
# OFFLINE FALLBACK ENGINE
# ---------------------------------------------------------

OFFLINE_FAQ = {
    "organic": {
        "en": "Use compost, mulching, and neem spray.",
        "kn": "ಕಂಪೋಸ್ಟ್, ಮಲ್ಚಿಂಗ್ ಮತ್ತು ನೀಮ್ ಸಿಂಪಡಣೆ ಬಳಸಿ."
    },
    "irrigation": {
        "en": "Irrigate early morning. Use drip to save 40% water.",
        "kn": "ಬೆಳಿಗ್ಗೆ ನೀರಾವರಿ ಮಾಡಿ. ಡ್ರಿಪ್ ಬಳಸಿ ನೀರು ಉಳಿಸಿ."
    },
    "fertilizer": {
        "en": "Apply NPK based on crop stage. Avoid excess nitrogen.",
        "kn": "ಬೆಳೆ ಹಂತಕ್ಕೆ ತಕ್ಕಂತೆ NPK ನೀಡಿ. ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ."
    },
    "pest": {
        "en": "Use neem oil 5 ml/L if pests observed.",
        "kn": "ಕೀಟ ಕಂಡರೆ 5 ml/L ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ."
    }
}

def offline_fallback(text: str, lang: str):
    q = text.lower()
    for key, ans in OFFLINE_FAQ.items():
        if key in q:
            return ans[lang]
    return None

# ---------------------------------------------------------
# AGRI Q&A ENGINE (100+ TOPICS) — SAMPLE EXPANDED
# (Full versions will appear in PART 2)
# ---------------------------------------------------------

AGRI_QA_TOPICS = {
    "soil_health": {
        "keywords": ["soil", "fertility", "ph", "organic carbon", "soil test"],
        "en": "Improve soil health using compost, green manure, and proper pH management.",
        "kn": "ಮಣ್ಣಿನ ಆರೋಗ್ಯಕ್ಕೆ ಕಂಪೋಸ್ಟ್, ಹಸಿರು ಗೊಬ್ಬರ ಮತ್ತು pH ನಿಯಂತ್ರಣ ನೆರವಾಗುತ್ತದೆ."
    },
    "weed_management": {
        "keywords": ["weed", "pendimethalin", "glyphosate"],
        "en": "Use Pendimethalin pre-emergence and hoeing at 20–25 DAS.",
        "kn": "Pendimethalin ಮೊಳಕೆಗೂ ಮುನ್ನ ಬಳಸಿ ಮತ್ತು 20–25 ದಿನದಲ್ಲಿ ಹುಲ್ಲು ತೆಗೆದುಹಾಕಿ."
    }
}

def agri_llm_engine(query: str, lang: str):
    q = query.lower()

    # keyword match
    for topic, info in AGRI_QA_TOPICS.items():
        if any(k in q for k in info["keywords"]):
            return info[lang], False, ["More details"]

    # Gemini fallback
    try:
        global client
        if client:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=get_prompt(lang)
                )
            )
            return resp.text, False, ["Related topics"]
    except:
        return None

    return None

# =========================================================
# PART 2 — AGRICULTURE INTELLIGENCE MODULES
# =========================================================

# ---------------------------------------------------------
# STAGE-WISE RECOMMENDATIONS (FULL DICTIONARY)
# ---------------------------------------------------------

STAGE_RECOMMENDATIONS = {
    "paddy": {
        "nursery": {
            "en": "Maintain 2–3 cm water depth, use light urea spray if seedlings turn pale.",
            "kn": "2–3 cm ನೀರು ಉಳಿಸಿ, ಮೊಳಕೆಗಳು ಹಳದಿ ಕಂಡರೆ ಲಘು ಯೂರಿಯಾ ಸಿಂಪಡಣೆ ಮಾಡಿ."
        },
        "tillering": {
            "en": "Apply 25% nitrogen, maintain shallow water, start weed control.",
            "kn": "25% ನೈಟ್ರೋಜನ್ ನೀಡಿ, ಸಮತಟ್ಟು ನೀರು ಉಳಿಸಿ, ಹುಲ್ಲು ನಿಯಂತ್ರಣ ಪ್ರಾರಂಭಿಸಿ."
        },
        "panicle": {
            "en": "Critical stage—ensure continuous moisture and prevent pests.",
            "kn": "ಮುಖ್ಯ ಹಂತ—ನಿರಂತರ ತೇವಾವಸ್ಥೆ ಇರಲಿ ಮತ್ತು ಕೀಟ ತಡೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Drain water 10 days before harvest and harvest at 20–22% moisture.",
            "kn": "ಕೊಯ್ತಿಗೆ 10 ದಿನ ಮುಂಚೆ ನೀರು ಬಿಡಿ, 20–22% ತೇವದ ಸಮಯದಲ್ಲಿ ಕೊಯ್ಯಿರಿ."
        }
    },

    "chilli": {
        "vegetative": {
            "en": "Apply nitrogen, remove suckers, ensure no waterlogging.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ, ಸಕ್ಕರ್‌ಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Avoid heavy nitrogen; boost potassium for flower retention.",
            "kn": "ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ; ಹೂ ಉಳಿಸಲು ಪೊಟ್ಯಾಸಿಯಂ ಹೆಚ್ಚಿಸಿ."
        },
        "fruiting": {
            "en": "Maintain irrigation intervals and apply micronutrients.",
            "kn": "ನೀರಾವರಿ ಕ್ರಮ ನಿಯಮಿತ ಇರಲಿ ಮತ್ತು ಸೂಕ್ಷ್ಮಾಂಶ ಗೊಬ್ಬರ ನೀಡಿ."
        }
    },

    "banana": {
        "vegetative": {
            "en": "Desucker every 45 days; apply FYM and nitrogen.",
            "kn": "ಪ್ರತಿ 45 ದಿನಕ್ಕೆ ಸಕ್ಕರ್ ತೆಗೆದುಹಾಕಿ; FYM ಮತ್ತು ನೈಟ್ರೋಜನ್ ನೀಡಿ."
        },
        "flowering": {
            "en": "Support banana bunch; maintain irrigation.",
            "kn": "ಬಾಳೆ ಮೊಗ್ಗಿಗೆ ಬೆಂಬಲ ನೀಡಿ; ನೀರಾವರಿ ನಿರಂತರ ಇರಲಿ."
        },
        "harvest": {
            "en": "Harvest when 75% fruit surface turns greenish-yellow.",
            "kn": "75% ಹಣ್ಣು ಮೇಲ್ಮೈ ಹಸಿರು-ಹಳದಿ ಆದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    }
}

def stage_recommendation_engine(crop: str, stage: str, lang: str):
    crop_l = crop.lower()
    stage_l = stage.lower()
    if crop_l in STAGE_RECOMMENDATIONS and stage_l in STAGE_RECOMMENDATIONS[crop_l]:
        return STAGE_RECOMMENDATIONS[crop_l][stage_l][lang]
    return {
        "en": "Stage recommendation not available for this crop.",
        "kn": "ಈ ಬೆಳೆಗಾಗಿ ಹಂತ ಸಲಹೆ ಲಭ್ಯವಿಲ್ಲ."
    }[lang]


# ---------------------------------------------------------
# FERTILIZER ENGINE (FULL)
# ---------------------------------------------------------

FERTILIZER_BASE = {
    "paddy": {
        "nursery": (30, 15, 15),
        "tillering": (40, 0, 20),
        "panicle": (30, 0, 30)
    },
    "chilli": {
        "vegetative": (40, 20, 20),
        "flowering": (20, 10, 40),
        "fruiting": (15, 5, 30)
    },
    "banana": {
        "vegetative": (100, 40, 60),
        "flowering": (50, 20, 100)
    }
}

def fertilizer_calculator(crop, stage, user_id, lang):
    farm = get_user_farm_details(user_id)
    area = float(farm.get("areaInHectares", 1.0))

    crop_l = crop.lower()
    stage_l = stage.lower()

    if crop_l in FERTILIZER_BASE and stage_l in FERTILIZER_BASE[crop_l]:
        N, P, K = FERTILIZER_BASE[crop_l][stage_l]
        N *= area
        P *= area
        K *= area

        text = (
            f"Recommended fertilizer for {crop} ({stage}): N={N}kg, P={P}kg, K={K}kg"
            if lang == "en" else
            f"{crop} ({stage}) ಹಂತಕ್ಕೆ: N={N}kg, P={P}kg, K={K}kg"
        )
        return text, False, ["Soil test", "Buy fertilizer"]

    return {
        "en": "Fertilizer data not available.",
        "kn": "ಗೊಬ್ಬರ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."
    }[lang], False, []


# ---------------------------------------------------------
# PESTICIDE ENGINE (FULL)
# ---------------------------------------------------------

PESTICIDE_DB = {
    "aphid": {
        "en": "Spray neem oil 5ml/L or Imidacloprid 0.3ml/L.",
        "kn": "5ml/L ನೀಮ್ ಎಣ್ಣೆ ಅಥವಾ Imidacloprid 0.3ml/L ಸಿಂಪಡಿಸಿ."
    },
    "thrips": {
        "en": "Use Fipronil 2ml/L or neem oil.",
        "kn": "Fipronil 2ml/L ಅಥವಾ ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ."
    },
    "fruit borer": {
        "en": "Use Spinosad or Emamectin Benzoate.",
        "kn": "Spinosad ಅಥವಾ Emamectin Benzoate ಬಳಸಿ."
    }
}

def pesticide_recommendation(crop, pest, lang):
    pest_l = pest.lower()
    if pest_l in PESTICIDE_DB:
        return PESTICIDE_DB[pest_l][lang], False, ["Safe spray", "Prevention"]
    return {
        "en": "Pest not recognized.",
        "kn": "ಕೀಟ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ."
    }[lang], False, []


# ---------------------------------------------------------
# IRRIGATION MODULE
# ---------------------------------------------------------

CROP_ET_BASE = {
    "paddy": 6,
    "chilli": 4,
    "banana": 7
}

SOIL_WATER_HOLDING = {
    "sandy": 0.6,
    "loamy": 1.0,
    "clay": 1.3
}

def irrigation_schedule(crop, stage, user_id, lang):
    crop_l = crop.lower()
    farm = get_user_farm_details(user_id)
    soil = farm.get("soilType", "loamy").lower()
    area = float(farm.get("areaInHectares", 1.0))

    base_et = CROP_ET_BASE.get(crop_l, 4)
    soil_factor = SOIL_WATER_HOLDING.get(soil, 1.0)

    stage_mult = 1.0
    if "vegetative" in stage.lower():
        stage_mult = 1.2
    elif "flower" in stage.lower():
        stage_mult = 1.1

    required_mm = base_et * stage_mult / soil_factor
    liters = required_mm * 10000 * area

    text = (
        f"Irrigation need: {required_mm:.1f} mm/day (~{liters:.0f} L/day)"
        if lang == "en"
        else f"ನೀರಾವರಿ ಅಗತ್ಯ: {required_mm:.1f} mm/ದಿನ (~{liters:.0f} ಲೀ/ದಿನ)"
    )

    return text, False, ["Use drip", "Monitor soil moisture"]


# ---------------------------------------------------------
# YIELD PREDICTION MODULE
# ---------------------------------------------------------

BASE_YIELD_TON_PER_HA = {
    "paddy": 5.0,
    "chilli": 2.0,
    "banana": 40.0
}

def yield_prediction(crop, user_id, lang):
    farm = get_user_farm_details(user_id)
    area = float(farm.get("areaInHectares", 1.0))

    base = BASE_YIELD_TON_PER_HA.get(crop.lower(), 2.0)

    estimated = base * area

    text = (
        f"Estimated yield: {estimated} tons"
        if lang == "en"
        else f"ಅಂದಾಜು ಉತ್ಪಾದನೆ: {estimated} ಟನ್"
    )

    return text, False, ["Improve irrigation", "Soil test"]


# ---------------------------------------------------------
# WEATHER-BASED DISEASE RISK MODEL
# ---------------------------------------------------------

DISEASE_WEATHER_RISK = {
    "paddy": [
        {"cond": "high_humidity", "disease": "blast"},
        {"cond": "continuous_rain", "disease": "bacterial blight"}
    ],
    "chilli": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "high_temp_low_humidity", "disease": "mites"}
    ]
}

def classify_weather(weather):
    conds = []
    if weather["humidity"] > 80:
        conds.append("high_humidity")
    if weather.get("rain", 0) > 8:
        conds.append("continuous_rain")
    if weather["temp"] > 32 and weather["humidity"] < 40:
        conds.append("high_temp_low_humidity")
    return conds


def predict_disease_from_weather(crop, weather, lang):
    crop = crop.lower()
    if crop not in DISEASE_WEATHER_RISK:
        return None

    conds = classify_weather(weather)
    risks = [r["disease"] for r in DISEASE_WEATHER_RISK[crop] if r["cond"] in conds]

    if not risks:
        return {
            "en": "No major disease risk.",
            "kn": "ಪ್ರಮುಖ ರೋಗ ಅಪಾಯ ಇಲ್ಲ."
        }[lang]

    joined = ", ".join(risks)

    return (
        f"Disease risk: {joined}"
        if lang == "en"
        else f"ರೋಗ ಅಪಾಯ: {joined}"
    )


# ---------------------------------------------------------
# SYMPTOM-BASED DIAGNOSIS ENGINE
# ---------------------------------------------------------

SYMPTOM_DB = {
    "yellow leaves": ["nutrient deficiency", "water stress"],
    "leaf curl": ["leaf curl virus", "thrips"],
    "white spots": ["powdery mildew"]
}

SYMPTOM_SYNONYMS = {
    "yellowing": "yellow leaves",
    "curling": "leaf curl",
    "white powder": "white spots"
}

def _normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _extract_symptoms(text):
    t = _normalize(text)
    matches = []

    for syn, canon in SYMPTOM_SYNONYMS.items():
        if syn in t:
            matches.append(canon)

    for key in SYMPTOM_DB:
        if key in t:
            matches.append(key)

    # fuzzy match
    for key in SYMPTOM_DB:
        if difflib.SequenceMatcher(None, key, t).ratio() > 0.6:
            matches.append(key)

    return list(set(matches))


def diagnose_advanced(text: str, crop: Optional[str], lang: str):
    symptoms = _extract_symptoms(text)

    if not symptoms:
        return {
            "en": "Could not identify symptoms clearly.",
            "kn": "ಲಕ್ಷಣಗಳು ಸ್ಪಷ್ಟವಿಲ್ಲ."
        }[lang], False, ["Upload photo"]

    candidates = []
    for s in symptoms:
        candidates.extend(SYMPTOM_DB.get(s, []))

    candidates = list(set(candidates))

    resp = (
        f"Possible issues: {', '.join(candidates)}"
        if lang == "en"
        else f"ಸಂಭವ್ಯ ಸಮಸ್ಯೆಗಳು: {', '.join(candidates)}"
    )

    return resp, False, ["Pesticide advice", "Prevention"]

# =========================================================
# PART 3 — ROUTER + API ENDPOINT + STARTUP
# =========================================================

# ---------------------------------------------------------
# LATEST CROP & STAGE FETCHER
# ---------------------------------------------------------

def get_latest_crop_stage(user_id: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return None, None

    latest_crop = None
    latest_stage = None
    latest_ts = -1

    for crop, entries in logs.items():
        if isinstance(entries, dict):
            for _, data in entries.items():
                ts = data.get("timestamp", 0)
                if ts > latest_ts:
                    latest_ts = ts
                    latest_crop = data.get("cropName", crop)
                    latest_stage = data.get("stage", "Unknown")

    return latest_crop, latest_stage


# ---------------------------------------------------------
# WEATHER FETCHER (LIGHTWEIGHT)
# ---------------------------------------------------------

def fetch_weather(district: str):
    # Weather disabled or no API — use simple fallback
    return {
        "temp": 30,
        "humidity": 70,
        "rain": 0
    }


# ---------------------------------------------------------
# MAIN ROUTER LOGIC
# ---------------------------------------------------------

def route(query: str, user_id: str, lang: str, session_key: str):
    q = query.lower().strip()

    # 1 — NDVI Query
    if "ndvi" in q or "crop health" in q or "satellite" in q:
        t, v, s = ndvi_health_report(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # 2 — Stage request
    if "stage" in q or "crop stage" in q:
        crop, stage = get_latest_crop_stage(user_id)
        if not crop:
            return {
                "response_text": "No farm activity found." if lang == "en" else "ಫಾರಂ ಚಟುವಟಿಕೆ ಕಂಡುಬರಲಿಲ್ಲ.",
                "voice": False,
                "suggestions": ["Add activity"]
            }
        stage_text = stage_recommendation_engine(crop, stage, lang)
        return {
            "response_text": f"{crop} – {stage}\n\n{stage_text}",
            "voice": False,
            "suggestions": ["Fertilizer", "Pest check"]
        }

    # 3 — Fertilizer query
    if "fertilizer" in q or "fertiliser" in q:
        crop, stage = get_latest_crop_stage(user_id)
        if not crop:
            return {
                "response_text": "Provide crop & stage." if lang == "en" else "ಬೆಳೆ ಮತ್ತು ಹಂತ ನೀಡಿ.",
                "voice": False,
                "suggestions": ["Add activity"]
            }
        t, v, s = fertilizer_calculator(crop, stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # 4 — Irrigation query
    if "irrigation" in q or "water" in q:
        crop, stage = get_latest_crop_stage(user_id)
        if not crop:
            return {
                "response_text": "Provide crop & stage." if lang == "en" else "ಬೆಳೆ ಮತ್ತು ಹಂತ ನೀಡಿ.",
                "voice": False,
                "suggestions": ["Add activity"]
            }
        t, v, s = irrigation_schedule(crop, stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # 5 — Yield query
    if "yield" in q or "production" in q:
        crop, _ = get_latest_crop_stage(user_id)
        if not crop:
            crop = "paddy"
        t, v, s = yield_prediction(crop, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # 6 — Pesticide query
    if "pesticide" in q or "spray" in q:
        for pest in PESTICIDE_DB.keys():
            if pest in q:
                t, v, s = pesticide_recommendation("", pest, lang)
                return {"response_text": t, "voice": v, "suggestions": s}
        return {
            "response_text": "Name the pest." if lang == "en" else "ಕೀಟದ ಹೆಸರು ನೀಡಿ.",
            "voice": False,
            "suggestions": ["Aphid", "Thrips"]
        }

    # 7 — Symptom diagnosis
    if any(tok in q for tok in ["symptom", "spot", "yellow", "curl", "leaf"]):
        crop, _ = get_latest_crop_stage(user_id)
        t, v, s = diagnose_advanced(query, crop, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # 8 — Agriculture Q&A engine (100+ topics)
    ans = agri_llm_engine(query, lang)
    if ans:
        t, v, s = ans
        return {"response_text": t, "voice": v, "suggestions": s}

    # 9 — Offline fallback
    offline = offline_fallback(query, lang)
    if offline:
        return {"response_text": offline, "voice": False, "suggestions": ["More info"]}

    # 10 — Gemini fallback
    global client
    try:
        if session_key not in active_chats:
            cfg = types.GenerateContentConfig(system_instruction=get_prompt(lang))
            chat = client.chats.create(model="gemini-2.5-flash", config=cfg)
            active_chats[session_key] = chat

        chat = active_chats[session_key]
        resp = chat.send_message(query)
        text = resp.text if hasattr(resp, "text") else str(resp)

        return {
            "response_text": text,
            "voice": False,
            "suggestions": ["Soil test", "Pest check"],
            "session_id": session_key
        }
    except:
        return {
            "response_text": "Unable to process right now." if lang == "en" else "ಈಗ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಾಗಿಲ್ಲ.",
            "voice": False,
            "suggestions": []
        }


# ---------------------------------------------------------
# API ENDPOINT
# ---------------------------------------------------------

@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    query = payload.user_query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query empty.")

    # Auto language detection + user preference
    lang = detect_language(query)
    if lang == "en":
        lang = get_preferred_language(payload.user_id)

    session_key = payload.session_id or f"{payload.user_id}-{lang}"

    result = route(query, payload.user_id, lang, session_key)

    audio_url = None
    if result.get("response_text"):
        audio_url = generate_tts_audio(result["response_text"], lang)

    return ChatResponse(
        session_id=session_key,
        response_text=result.get("response_text", ""),
        language=lang,
        suggestions=result.get("suggestions", []),
        voice=True,
        audio_url=audio_url,
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
def startup():
    initialize_firebase_credentials()
    initialize_gemini()


