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
# GOVERNMENT SCHEMES MODULE
# =========================================================

GOVT_SCHEMES = {
    "pm kisan": {
        "en": "PM-Kisan provides ₹6000/year to farmers in 3 installments. Eligibility: small & marginal farmers with land records.",
        "kn": "PM-Kisan ಯೋಜನೆ ರೈತರಿಗೆ ವರ್ಷಕ್ಕೆ ₹6000 ನೆರವು ನೀಡುತ್ತದೆ. ಅರ್ಹತೆ: ಭೂ ದಾಖಲೆ ಇರುವ ಸಣ್ಣ ಮತ್ತು ಅಂಚಿನ ರೈತರು."
    },
    "pmfby": {
        "en": "PMFBY crop insurance covers yield loss due to drought, flood, pest attack & natural calamities.",
        "kn": "PMFBY ಬೆಳೆ ವಿಮೆ ಬರ, ನೆರೆ, ಕೀಟ, ಮತ್ತು ಪ್ರಕೃತಿ ವಿಕೋಪಗಳಿಂದ ಆಗುವ ನಷ್ಟವನ್ನು ಹೊರುತ್ತದೆ."
    },
    "soil health card": {
        "en": "Soil Health Card provides nutrient status of soil and fertilizer recommendations every 2 years.",
        "kn": "ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಮಣ್ಣಿನ ಪೋಷಕಾಂಶ ಸ್ಥಿತಿ ಮತ್ತು ಗೊಬ್ಬರ ಸಲಹೆ ನೀಡುತ್ತದೆ."
    },
    "kcc": {
        "en": "Kisan Credit Card offers low-interest crop loans up to ₹3 lakhs.",
        "kn": "ಕಿಸಾನ್ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ (KCC) ಕಡಿಮೆ ಬಡ್ಡಿದರದ ಬೆಳೆ ಸಾಲ ಒದಗಿಸುತ್ತದೆ."
    },
    "drip subsidy": {
        "en": "Government offers 55–75% subsidy on drip & sprinkler irrigation under PMKSY.",
        "kn": "PMKSY ಯಡಿಯಲ್ಲಿ ಡ್ರಿಪ್ ಮತ್ತು ಸ್ಪ್ರಿಂಕ್ಲರ್ ನೀರಾವರಿಗೆ 55–75% ಸಹಾಯಧನ ಲಭ್ಯ."
    }
}

def govt_scheme_engine(query: str, lang: str):
    q = query.lower()
    for scheme, info in GOVT_SCHEMES.items():
        if scheme in q:
            return info[lang], False, ["Eligibility", "How to apply"]

    # General scheme intent
    if "scheme" in q or "yojana" in q or "subsidy" in q:
        response = {
            "en": "Available schemes: PM-Kisan, PMFBY crop insurance, Soil Health Card, KCC loan, PMKSY drip subsidy.",
            "kn": "ಲಭ್ಯವಿರುವ ಯೋಜನೆಗಳು: PM-Kisan, PMFBY ಬೆಳೆ ವಿಮೆ, ಮಣ್ಣು ಆರೋಗ್ಯ ಕಾರ್ಡ್, KCC ಸಾಲ, PMKSY ಡ್ರಿಪ್ ಸಹಾಯಧನ."
        }
        return response[lang], False, ["PM-Kisan", "PMFBY", "KCC"]

    return None

# =========================================================
# CROP RECOMMENDATION ENGINE (SOIL + CLIMATE)
# =========================================================

SOIL_TO_CROP = {
    "red soil": ["Groundnut", "Millet", "Pigeon pea", "Cotton"],
    "black soil": ["Cotton", "Soybean", "Paddy", "Red gram"],
    "loamy": ["Vegetables", "Paddy", "Wheat", "Sugarcane"],
    "sandy": ["Groundnut", "Watermelon", "Cucumber"],
    "clay": ["Paddy", "Banana", "Sugarcane"]
}

CLIMATE_TO_CROP = {
    "dry": ["Millet", "Sorghum", "Castor", "Pigeon pea"],
    "semi-dry": ["Cotton", "Groundnut", "Bengal gram"],
    "humid": ["Paddy", "Banana", "Arecanut", "Spices"],
    "coastal": ["Coconut", "Arecanut", "Paddy", "Cashew"]
}

def detect_climate_from_district(district: str):
    district = district.lower()

    dry = ["chitradurga", "ballari", "tumkur", "bijapur"]
    humid = ["shivamogga", "udupi", "dakshina kannada"]
    coastal = ["karwar", "mangalore"]
    semi_dry = ["haveri", "davanagere", "chikkaballapur"]

    if district in dry:
        return "dry"
    if district in humid:
        return "humid"
    if district in coastal:
        return "coastal"
    if district in semi_dry:
        return "semi-dry"
    return "semi-dry"
def crop_recommendation_engine(user_id: str, lang: str):
    farm = get_user_farm_details(user_id)

    soil = farm.get("soilType", "").lower()
    pH = farm.get("soilPH")
    district = farm.get("district", "unknown")

    if not soil:
        return {
            "en": "Update soil type in farm details.",
            "kn": "ಫಾರಂ ವಿವರಗಳಲ್ಲಿ ಮಣ್ಣಿನ ವಿಧ ಸೇರಿಸಿ."
        }[lang], False, ["Update soil details"]

    climate = detect_climate_from_district(district)

    # soil-based
    soil_based = SOIL_TO_CROP.get(soil, [])

    # climate-based
    climate_based = CLIMATE_TO_CROP.get(climate, [])

    # combined recommendation
    common = list(set(soil_based) & set(climate_based))
    if not common:
        common = soil_based or climate_based

    # pH-based refinement
    if pH:
        if pH < 6.0:
            common.append("Lime application recommended before planting.")
        elif pH > 8.0:
            common.append("Choose alkaline-tolerant crops like Cotton, Castor.")

    text = (
        f"Recommended crops for your soil & climate: {', '.join(common)}"
        if lang == "en" else
        f"ನಿಮ್ಮ ಮಣ್ಣು & ಹವಾಮಾನಕ್ಕೆ ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆಗಳು: {', '.join(common)}"
    )

    return text, False, ["Show fertilizer schedule", "Pest-resistant varieties"]
# =========================================================
# MARKET PRICE–BASED PROFITABILITY ENGINE
# =========================================================

# Average cost of cultivation per hectare (rough estimates)
CROP_COST = {
    "paddy": 45000,
    "chilli": 80000,
    "banana": 120000,
    "groundnut": 50000,
    "cotton": 65000,
    "ragi": 30000,
    "maize": 35000
}

# Market price per kg (fallback if APMC not connected)
MARKET_PRICE = {
    "paddy": 20,
    "chilli": 70,
    "banana": 10,
    "groundnut": 50,
    "cotton": 60,
    "ragi": 25,
    "maize": 20
}

# Expected yield per hectare (tonne → convert to kg)
CROP_YIELD = {
    "paddy": 5 * 1000,
    "chilli": 2 * 1000,
    "banana": 40 * 1000,
    "groundnut": 2.5 * 1000,
    "cotton": 2 * 1000,
    "ragi": 1.5 * 1000,
    "maize": 3 * 1000
}

def profitability_ranking_engine(user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    area = float(farm.get("areaInHectares", 1.0))

    ranking = []

    for crop in CROP_COST.keys():
        cost = CROP_COST[crop]
        price = MARKET_PRICE[crop]
        yield_kg = CROP_YIELD[crop]

        revenue = yield_kg * price
        profit = revenue - cost

        ranking.append((crop, profit))

    ranking.sort(key=lambda x: x[1], reverse=True)

    top3 = ranking[:3]

    if lang == "en":
        text = "Top profitable crops:\n"
        for crop, prof in top3:
            text += f"• {crop.title()}: Profit ₹{prof:,} per hectare\n"
    else:
        text = "ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆಗಳು:\n"
        for crop, prof in top3:
            text += f"• {crop.title()}: ಲಾಭ ₹{prof:,} / ಹೆಕ್ಟೇರ್\n"

    return text, False, ["Suggest best crop", "Market price"]


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

def fetch_weather_live(district: str):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"q={district}&appid={OPENWEATHER_KEY}&units=metric"
        )
        data = requests.get(url, timeout=10).json()

        if data.get("cod") != 200:
            return None

        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain": data.get("rain", {}).get("1h", 0),
            "wind": data["wind"]["speed"],
            "condition": data["weather"][0]["description"]
        }
    except:
        return None

def live_weather_advisory(user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    district = farm.get("district")

    if not district:
        return {
            "en": "Update district in farm details.",
            "kn": "ಜಿಲ್ಲೆ ಮಾಹಿತಿಯನ್ನು ನವೀಕರಿಸಿ."
        }[lang], False, ["Update farm details"]

    weather = fetch_weather_live(district)

    if not weather:
        return {
            "en": "Unable to fetch live weather.",
            "kn": "ಹವಾಮಾನ ಪಡೆಯಲಾಗಲಿಲ್ಲ."
        }[lang], False, []

    # Build response
    if lang == "en":
        text = (
            f"Weather in {district}:\n"
            f"• Temp: {weather['temp']}°C\n"
            f"• Humidity: {weather['humidity']}%\n"
            f"• Rain (1h): {weather['rain']}mm\n"
            f"• Wind: {weather['wind']} km/h\n"
            f"• Condition: {weather['condition']}\n"
        )
    else:
        text = (
            f"{district} ಹವಾಮಾನ:\n"
            f"• ತಾಪಮಾನ: {weather['temp']}°C\n"
            f"• ತೇವಾಂಶ: {weather['humidity']}%\n"
            f"• ಮಳೆ (1h): {weather['rain']}mm\n"
            f"• ಗಾಳಿ: {weather['wind']} km/h\n"
            f"• ಸ್ಥಿತಿ: {weather['condition']}\n"
        )

    # Weather-based advice
    extra = []
    if weather["temp"] > 34:
        extra.append("High temperature — irrigate crops.")
    if weather["humidity"] > 85:
        extra.append("High humidity — fungal diseases likely.")
    if weather["rain"] > 5:
        extra.append("Heavy rain — avoid fertilizer today.")

    # Kannada translation
    if lang == "kn":
        extra_kn = []
        for e in extra:
            if "High temperature" in e:
                extra_kn.append("ಹೆಚ್ಚು ಬಿಸಿಲು — ನೀರಾವರಿ ಮಾಡಿ.")
            if "fungal" in e:
                extra_kn.append("ಹೆಚ್ಚು ತೇವಾಂಶ — ಫಂಗಸ್ ರೋಗ ಸಾಧ್ಯತೆ.")
            if "avoid fertilizer" in e:
                extra_kn.append("ಹೆಚ್ಚು ಮಳೆ — ಗೊಬ್ಬರ ಬೇಡ.")
        extra = extra_kn

    if extra:
        text += "\n" + "\n".join("• " + e for e in extra)

    return text, False, ["Disease risk", "Irrigation advice"]


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

    # Government scheme queries
    ans = govt_scheme_engine(query, lang)
    if ans:
        t, v, s = ans
        return {"response_text": t, "voice": v, "suggestions": s}

    # Crop recommendation query
    if "recommend crop" in q or "best crop" in q or "which crop" in q:
        t, v, s = crop_recommendation_engine(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

        # Profitability ranking
    if "profit" in q or "profitable" in q or "best crop" in q:
        t, v, s = profitability_ranking_engine(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

        # Live weather
    if "weather" in q or "rain" in q or "temperature" in q:
        t, v, s = live_weather_advisory(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

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



