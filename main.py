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

    # ---------------------------------------------------
    # SOIL HEALTH
    # ---------------------------------------------------
    "soil_health": {
        "keywords": ["soil", "fertility", "ph", "organic carbon", "soil test", "salinity"],
        "en": "Improve soil health using compost, green manure, crop rotation, mulching, and regular soil testing.",
        "kn": "ಮಣ್ಣಿನ ಆರೋಗ್ಯಕ್ಕಾಗಿ ಕಂಪೋಸ್ಟ್, ಹಸಿರು ಗೊಬ್ಬರ, ಬೆಳೆ ಪರಿವರ್ತನೆ, ಮಲ್ಚಿಂಗ್ ಮತ್ತು ನಿಯಮಿತ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಅತ್ಯವಶ್ಯಕ."
    },
    "soil_ph_management": {
        "keywords": ["ph", "acidic", "alkaline", "lime", "dolomite"],
        "en": "For acidic soils apply lime/dolomite. For alkaline soils use organic matter and gypsum.",
        "kn": "ಆಮ್ಲೀಯ ಮಣ್ಣಿಗೆ ಲೈಮ್/ಡೋಲೊಮೈಟ್ ಬಳಸಿ. ಕ್ಷಾರ ಮಣ್ಣಿಗೆ ಜಿಪ್ಸಮ್ ಮತ್ತು ಜೈವಿಕ ಗೊಬ್ಬರ ಬಳಸಿ."
    },
    "soil_micronutrients": {
        "keywords": ["zinc", "boron", "micronutrient", "deficiency"],
        "en": "Zinc deficiency: apply Zinc Sulphate. Boron deficiency: apply Borax 1–2 kg/acre.",
        "kn": "ಜಿಂಕ್ ಕೊರತೆ: Zinc Sulphate ಬಳಸಿ. ಬೋರಾನ್ ಕೊರತೆ: Borax 1–2 kg/acre ಬಳಸಿ."
    },

    # ---------------------------------------------------
    # WEED MANAGEMENT
    # ---------------------------------------------------
    "weed_management": {
        "keywords": ["weed", "pendimethalin", "glyphosate", "pre-emergence"],
        "en": "Use Pendimethalin as pre-emergence and hand weeding at 20–25 DAS.",
        "kn": "Pendimethalin ಮೊಳಕೆಗೂ ಮುನ್ನ ಬಳಸಿ ಮತ್ತು 20–25 ದಿನದಲ್ಲಿ ಕಳೆ ತೆಗೆದುಹಾಕಿ."
    },
    "post_emergence_herbicides": {
        "keywords": ["post emergence", "quizalofop", "2,4-d", "herbicide"],
        "en": "Use Quizalofop for grassy weeds and 2,4-D for broadleaf weeds.",
        "kn": "Quizalofop ಅನ್ನು ಹುಲ್ಲು ಕಳೆಗಳಿಗೆ ಮತ್ತು 2,4-D ಅನ್ನು ಅಗಲ ಎಲೆ ಕಳೆಗಳಿಗೆ ಬಳಸಿ."
    },

    # ---------------------------------------------------
    # IRRIGATION & WATER MANAGEMENT
    # ---------------------------------------------------
    "irrigation_basics": {
        "keywords": ["irrigation", "water", "drip", "sprinkler"],
        "en": "Follow drip irrigation for water saving and better nutrient use efficiency.",
        "kn": "ನೀರಿನ ಉಳಿತಾಯ ಹಾಗೂ ಉತ್ತಮ ಪೋಷಕಾಂಶ ಬಳಕೆಗಾಗಿ ಡ್ರಿಪ್ ನೀರಾವರಿ ಅನುಸರಿಸಿ."
    },
    "rainwater_harvesting": {
        "keywords": ["rainwater", "farm pond", "harvest water"],
        "en": "Build farm ponds and bunding structures to store rainwater.",
        "kn": "ಮಳಿನೀರನ್ನು ಸಂಗ್ರಹಿಸಲು ಫಾರಂ ಪಾಂಡ್ ಮತ್ತು ಬಂಡಿಂಗ್ ರಚನೆಗಳನ್ನು ನಿರ್ಮಿಸಿ."
    },
    "drought_management": {
        "keywords": ["drought", "dry spell"],
        "en": "Use mulching, reduce fertilizer, and irrigate lightly during drought.",
        "kn": "ಕರಡು ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಮಲ್ಚಿಂಗ್ ಮಾಡಿ, ಗೊಬ್ಬರ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಸ್ವಲ್ಪ ನೀರಾವರಿ ಮಾಡಿ."
    },

    # ---------------------------------------------------
    # FERTILIZER & NUTRIENT MANAGEMENT
    # ---------------------------------------------------
    "fertilizer_basics": {
        "keywords": ["fertilizer", "urea", "dap", "npk"],
        "en": "Apply fertilizers based on soil test. Split nitrogen for better efficiency.",
        "kn": "ಮಣ್ಣು ಪರೀಕ್ಷೆಯ ಆಧಾರದಲ್ಲಿ ಗೊಬ್ಬರ ನೀಡಿ. ನೈಟ್ರೋಜನ್ ಅನ್ನು ಹಂತ ಹಂತವಾಗಿ ನೀಡಿ."
    },
    "organic_fertilizers": {
        "keywords": ["organic", "fym", "compost", "vermicompost"],
        "en": "Use FYM, compost, and vermicompost to improve soil structure and microbial activity.",
        "kn": "FYM, ಕಂಪೋಸ್ಟ್ ಮತ್ತು ವರ್ಮಿಕಂಪೋಸ್ಟ್ ಮಣ್ಣಿನ ಗುಣಾತ್ಮಕತೆಯನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ."
    },

    # ---------------------------------------------------
    # PEST & DISEASE GENERAL ADVICE
    # ---------------------------------------------------
    "ipm_basics": {
        "keywords": ["ipm", "integrated pest", "manage pests"],
        "en": "Use IPM: crop rotation, pheromone traps, neem oil, and need-based sprays.",
        "kn": "IPM ಅನುಸರಿಸಿ: ಬೆಳೆ ಪರಿವರ್ತನೆ, ಫೆರಮೊನ್ ಟ್ರಾಪ್, ನೀಮ್ ಎಣ್ಣೆ ಮತ್ತು ಅಗತ್ಯವಿದ್ದಾಗ ಮಾತ್ರ ಸಿಂಪಡಣೆ."
    },
    "fungal_diseases": {
        "keywords": ["fungus", "blight", "powdery", "mildew"],
        "en": "Ensure aeration, avoid overcrowding, and use preventive fungicides.",
        "kn": "ಗಾಳಿ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ, ಬೆಳೆ ಗಿಡಗಳ ದಟ್ಟಣೆ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಫಂಗಿಸೈಡ್‌ಗಳನ್ನು ಮುನ್ನೆಚ್ಚರಿಕಾ ಕ್ರಮವಾಗಿ ಬಳಸಿ."
    },

    # ---------------------------------------------------
    # SEED & SOWING
    # ---------------------------------------------------
    "seed_treatment": {
        "keywords": ["seed treatment", "trichoderma", "bavistin"],
        "en": "Treat seeds with Trichoderma or Bavistin before sowing.",
        "kn": "ಬಿತ್ತನೆಗೂ ಮೊದಲು ಬೀಜಗಳನ್ನು Trichoderma ಅಥವಾ Bavistin ನಿಂದ ಶೋಧಿಸಿ."
    },
    "sowing_time": {
        "keywords": ["sowing", "when to sow", "timing"],
        "en": "Sow according to local rainfall pattern and recommended crop calendar.",
        "kn": "ಬಿತ್ತನೆ ಸ್ಥಳೀಯ ಮಳೆಯ ಮಾದರಿ ಮತ್ತು ಬೆಳೆ ಕ್ಯಾಲೆಂಡರ್ ಪ್ರಕಾರ ಮಾಡಿ."
    },

    # ---------------------------------------------------
    # CROP MANAGEMENT
    # ---------------------------------------------------
    "crop_rotation": {
        "keywords": ["crop rotation", "alternate crops"],
        "en": "Rotate cereals with legumes to improve soil nitrogen.",
        "kn": "ಧಾನ್ಯಗಳೊಂದಿಗೆ ಕಾಳುಬೆಳೆಗಳನ್ನು ಪರ್ಯಾಯವಾಗಿ ಬೆಳೆಸಿ ನೈಟ್ರೋಜನ್ ಹೆಚ್ಚಿಸಿ."
    },
    "mulching": {
        "keywords": ["mulch", "plastic mulch", "straw mulch"],
        "en": "Mulching reduces evaporation, weeds, and improves soil health.",
        "kn": "ಮಲ್ಚಿಂಗ್ ಮಣ್ಣಿನ ತೇವವನ್ನು ಉಳಿಸಿ, ಕಳೆ ಕಡಿಮೆ ಮಾಡಿ ಹಾಗೂ ಮಣ್ಣಿನ ಗುಣ ಹೆಚ್ಚಿಸುತ್ತದೆ."
    },
    "pruning_training": {
        "keywords": ["pruning", "training", "branch cutting"],
        "en": "Pruning improves aeration, sunlight penetration, and fruit size.",
        "kn": "ಪ್ರುನಿಂಗ್ ಗಾಳಿ ಸಂಚಾರ, ಬೆಳಕಿನ ಪ್ರವೇಶ ಮತ್ತು ಹಣ್ಣಿನ ಗಾತ್ರ ಹೆಚ್ಚಿಸುತ್ತದೆ."
    },

    # ---------------------------------------------------
    # HARVEST & POST-HARVEST
    # ---------------------------------------------------
    "harvest_time": {
        "keywords": ["harvest", "maturity", "ripening"],
        "en": "Harvest crops at proper maturity to get maximum yield and quality.",
        "kn": "ಅತ್ಯುತ್ತಮ ಫಲ ಮತ್ತು ಗುಣಮಟ್ಟಕ್ಕಾಗಿ ಸಮರ್ಪಕ ಪ್ರಾಪ್ತಿಯ ವೇಳೆಯಲ್ಲಿ ಬೆಳೆ ಕೊಯ್ಯಿರಿ."
    },
    "storage_loss": {
        "keywords": ["storage", "warehouse", "grain moisture"],
        "en": "Dry grains to 12–13% moisture and store in airtight bins to avoid pests.",
        "kn": "ಧಾನ್ಯಗಳನ್ನು 12–13% ತೇವಕ್ಕೆ ಒಣಗಿಸಿ ಗಾಳಿಯೂಡುವ ಮದ್ದುಪಾತ್ರೆಗಳಲ್ಲಿ ಸಂಗ್ರಹಿಸಿ."
    },

    # ---------------------------------------------------
    # FARM ECONOMICS
    # ---------------------------------------------------
    "farm_budgeting": {
        "keywords": ["budget", "cost", "profit"],
        "en": "Prepare a crop-wise budget including seeds, fertilizer, labour, irrigation.",
        "kn": "ಬೀಜ, ಗೊಬ್ಬರ, ಕಾರ್ಮಿಕ, ನೀರಾವರಿ ಇತ್ಯಾದಿಗಳನ್ನು ಒಳಗೊಂಡಂತೆ ಬೆಳೆ ಆಧಾರದ ಬಜೆಟ್ ತಯಾರಿಸಿ."
    },
    "market_strategy": {
        "keywords": ["market", "selling", "price", "mandi"],
        "en": "Check multiple mandis, avoid distress selling, and grade produce.",
        "kn": "ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಹೋಲಿಸಿ, ಒತ್ತಡದ ಮಾರಾಟ ತಪ್ಪಿಸಿ ಮತ್ತು ಉತ್ಪನ್ನ ಗ್ರೇಡಿಂಗ್ ಮಾಡಿ."
    },

    # ---------------------------------------------------
    # SUSTAINABLE / CLIMATE SMART FARMING
    # ---------------------------------------------------
    "climate_smart": {
        "keywords": ["climate", "smart", "carbon", "emission"],
        "en": "Use drip irrigation, crop rotation, and reduced tillage for climate-smart farming.",
        "kn": "ಡ್ರಿಪ್ ನೀರಾವರಿ, ಬೆಳೆ ಪರಿವರ್ತನೆ ಮತ್ತು ಕಡಿಮೆ ಜೋತೆ ಕೆಲಸ ಹವಾಮಾನ ಸ್ನೇಹಿ ಕೃಷಿ ಕ್ರಮಗಳು."
    },
    "carbon_farming": {
        "keywords": ["carbon", "sequestration", "biochar"],
        "en": "Use biochar, mulching, and agroforestry to increase soil carbon.",
        "kn": "ಬಯೋಚಾರ್, ಮಲ್ಚಿಂಗ್ ಮತ್ತು ಅಗ್ರೊಫಾರೆಸ್ಟ್ರಿ ಮಣ್ಣಿನ ಕಾರ್ಬನ್ ಹೆಚ್ಚಿಸುತ್ತದೆ."
    },

    # ---------------------------------------------------
    # GREENHOUSE & ADVANCED FARMING
    # ---------------------------------------------------
    "greenhouse_basics": {
        "keywords": ["polyhouse", "greenhouse", "shade net"],
        "en": "Greenhouses allow off-season production and higher yield for vegetables & flowers.",
        "kn": "ಗ್ರೀನ್ಹೌಸ್ ತರಕಾರಿ ಮತ್ತು ಹೂಗಳ ಹೆಚ್ಚಿನ ಉತ್ಪಾದನೆಯನ್ನು ನೀಡುತ್ತದೆ."
    },
    "hydroponics": {
        "keywords": ["hydroponic", "soilless"],
        "en": "Hydroponics uses nutrient solution instead of soil for faster production.",
        "kn": "Hydroponics ಮಣ್ಣಿನ ಬದಲಿಗೆ ಪೋಷಕ ದ್ರಾವಣ ಬಳಸಿ ವೇಗವಾದ ಉತ್ಪಾದನೆ ನೀಡುತ್ತದೆ."
    },

    # ---------------------------------------------------
    # LIVESTOCK (BASIC)
    # ---------------------------------------------------
    "cow_management": {
        "keywords": ["cow", "dairy", "milk"],
        "en": "Provide balanced feed, clean water, and regular deworming.",
        "kn": "ಸಮತೂಲ್ಯ ಆಹಾರ, ಸ್ವಚ್ಛ ನೀರು ಮತ್ತು ನಿಯಮಿತ ಡಿವರ್ಮಿಂಗ್ ಅಗತ್ಯ."
    },
    "goat_farming": {
        "keywords": ["goat", "buck", "doe"],
        "en": "Goat farming requires good housing, mineral mixture, and vaccination.",
        "kn": "ಮೇಯಲು ಸ್ಥಳ, ಮಿನರಲ್ ಮಿಶ್ರಣ ಮತ್ತು ಲಸಿಕೆ ಗೋವು ಸಾಕಣೆಗೆ ಮುಖ್ಯ."
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
                "en": "Use 20–25 kg seed/ha in dry season; maintain 2–3 cm water; treat seeds with fungicide; apply light urea if seedlings pale.",
                "kn": "ಒಣಗಾಲದಲ್ಲಿ 20–25 ಕೆಜಿ ಬೀಜ/ಹೆಕ್ಟೇರ್ ಬಳಸಿ; 2–3 ಸೆಂ.ಮೀ ನೀರು ಉಳಿಸಿ; ಬಿತ್ತನೆಗೂ ಮುನ್ನ ಬೀಜ ಶೋಧಿಸಿ; ಮೊಳಕೆಗಳು ಹಳದಿ ಕಂಡರೆ ಲಘು ಯೂರಿಯಾ ನೀಡಿ."
        },
            "vegetative": {
                "en": "Maintain 3–5 cm water; apply first split of nitrogen; remove weeds using cono weeder.",
                "kn": "3–5 ಸೆಂ.ಮೀ ನೀರು ಕಾಯ್ದುಕೊಳ್ಳಿ; ಮೊದಲ ಹಂತದ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕೊನೋ ವೀಡರ್ ಬಳಸಿ ಕಳೆ ತೆಗೆದುಹಾಕಿ."
        },
            "tillering": {
                "en": "Apply 25% nitrogen; control weeds; maintain shallow water; prevent stem borer.",
                "kn": "25% ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ನಿಯಂತ್ರಣ ಮಾಡಿ; ಸಮತಟ್ಟು ನೀರು ಇರಲಿ; ಸ್ಟೆಮ್ ಬೋರರ್ ತಡೆ ಕ್ರಮ ಅನುಸರಿಸಿ."
        },
            "panicle": {
                "en": "Critical stage; ensure moisture; apply potassium; avoid heavy nitrogen; monitor for neck blast.",
                "kn": "ಮುಖ್ಯ ಹಂತ; ತೇವಾವಸ್ಥೆ ಉಳಿಸಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ; ನೆಕ್ ಬ್ಲಾಸ್ಟ್ ತಡೆ ಕ್ರಮ ಜಾರಿಗೆ ತರು."
        },
            "maturity": {
                "en": "Reduce irrigation; maintain field dryness; avoid lodging by reducing nitrogen.",
                "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಹೊಲ ಒಣವಾಗಿಡಿ; ನೈಟ್ರೋಜನ್ ಕಡಿಮೆ ಮಾಡಿ ಲಾಡ್ಜಿಂಗ್ ತಪ್ಪಿಸಿ."
        },
            "harvest": {
                "en": "Drain water 10 days before harvest; harvest at 20–22% grain moisture.",
                "kn": "ಕೊಯ್ತಿಗೆ 10 ದಿನ ಮೊದಲು ನೀರು ಬಿಡಿ; 20–22% ಧಾನ್ಯ ತೇವದ ಮಟ್ಟದಲ್ಲಿ ಕೊಯ್ಯಿರಿ."
        }
    },


    "ragi": {
        "nursery": {
            "en": "Use well-prepared raised beds; apply FYM; maintain light moisture; protect from leaf blast.",
            "kn": "ಉತ್ತಮ ಎತ್ತಿದ ಮಂಚներում ಬಿತ್ತನೆ ಮಾಡಿ; FYM ನೀಡಿ; ಲಘು ತೇವ ಇರಲಿ; ಎಲೆ ಬ್ಲಾಸ್ಟ್ ತಡೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Thin seedlings 15–20 days after sowing; apply urea; maintain moderate moisture.",
            "kn": "ಬಿತ್ತನೆ ನಂತರ 15–20 ದಿನಕ್ಕೆ ತೆನೆ ತೆಗೆಯಿರಿ; ಯೂರಿಯಾ ನೀಡಿ; ಮಧ್ಯಮ ತೇವ ಇರಲಿ."
        },
        "tillering": {
            "en": "Second dose nitrogen; keep field weed-free; ensure adequate sunlight.",
            "kn": "ಎರಡನೇ ಹಂತದ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಸಾಕಷ್ಟು ಬೆಳಕು ಒದಗಿಸಿ."
        },
        "panicle": {
            "en": "Apply potash for finger strengthening; avoid water stress.",
            "kn": "ಎಲುಬು ಬಲಕ್ಕಾಗಿ ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಿನ ಕೊರತೆ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Reduce irrigation; ensure no lodging; protect from birds.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಲಾಡ್ಜಿಂಗ್ ತಪ್ಪಿಸಿ; ಹಕ್ಕಿಗಳಿಂದ ರಕ್ಷಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when earheads turn brown; dry before storage.",
            "kn": "ಮುಳ್ಳುಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಸಂಗ್ರಹಣೆಗೆ ಮುಂಚೆ ಒಣಗಿಸಿ."
        }
    },


    "jowar": {
        "nursery": {
            "en": "Direct sowing preferred; seed rate 8–10 kg/ha; treat seeds with fungicide.",
            "kn": "ನೆರಳಾಗಿ ಬಿತ್ತನೆ ಉತ್ತಮ; 8–10 ಕೆಜಿ ಬೀಜ/ಹೆ; ಬೀಜ ಶೋಧನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen at 20–25 DAS; first intercultivation; maintain optimum spacing.",
            "kn": "20–25 DAS ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮೊದಲ ಮಧ್ಯಂತರ ಕಳಿತೋಡು; ಸರಿಯಾದ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "tillering": {
            "en": "Second top dressing; remove weeds; check for shoot fly.",
            "kn": "ಎರಡನೇ ಹಂತದ ಗೊಬ್ಬರ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಶೂಟ್ ಫ್ಲೈ ತಡೆ ಮಾಡಿ."
        },
        "panicle": {
            "en": "Ensure moisture; avoid drought; protect from grain midge.",
            "kn": "ತೇವ ಇರಲಿ; ಬರ ತಪ್ಪಿಸಿ; ಗ್ರೇನ್ ಮಿಡ್ಜ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Reduce watering; avoid lodging; grain hardening stage.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಲಾಡ್ಜಿಂಗ್ ತಪ್ಪಿಸಿ; ಧಾನ್ಯ ಗಟ್ಟಿಯಾಗುವ ಹಂತ."
        },
        "harvest": {
            "en": "Harvest when grains become hard and shiny.",
            "kn": "ಧಾನ್ಯ ಗಟ್ಟಿ ಮತ್ತು ಹೊಳೆಯುವಾಗ ಕೊಯ್ತಿಗೆ ತರು."
        }
    },
    "maize": {
        "nursery": {
            "en": "Direct sowing; treat seeds with fungicide/insecticide; ensure good soil tilth.",
            "kn": "ನೆರಳಾಗಿ ಬಿತ್ತನೆ ಮಾಡಿ; ಬೀಜ ಶೋಧನೆ ಮಾಡಿ; ಮಣ್ಣು ಸೊಂಪಾಗಿರಲಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen at knee-high stage; ensure good sunlight; control weeds by intercultivation.",
            "kn": "ಕಾಲ್ಮೂಳೆ ಹಂತದಲ್ಲಿ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಬೆಳಕು ಸಾಕಷ್ಟು; ಮಧ್ಯಂತರ ಕಳಿತೋಡು ಮಾಡಿ."
        },
        "tillering": {
            "en": "Top dress urea; maintain moisture; check for stem borer.",
            "kn": "ಟಾಪ್ ಡ್ರೆಸ್ ಯೂರಿಯಾ ನೀಡಿ; ತೇವ ಇರಲಿ; ಸ್ಟೆಮ್ ಬೋರರ್ ತಡೆ ನೋಡಿ."
        },
        "panicle": {
            "en": "Critical water stage; avoid moisture stress; apply potash.",
            "kn": "ಮುಖ್ಯ ನೀರಾವರಿ ಹಂತ; ನೀರಿನ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "maturity": {
            "en": "Drying of lower leaves normal; reduce irrigation.",
            "kn": "ಕೆಳ ಎಲೆಗಳ ಒಣಗುವುದು ಸಹಜ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest cobs when kernels are hard and dry.",
            "kn": "ದಾಣೆಗಳು ಗಟ್ಟಿ, ಒಣವಾಗಿದ್ದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
    "bajra": {
        "nursery": {
            "en": "Use 3–4 kg seed/ha; sow on ridges; seed treatment essential.",
            "kn": "3–4 ಕೆಜಿ ಬೀಜ/ಹೆ; ಎತ್ತರದ ರಿಡ್ಜ್ ಮೇಲೆ ಬಿತ್ತನೆ; ಬೀಜ ಶೋಧನೆ ಅಗತ್ಯ."
        },
        "vegetative": {
            "en": "Apply urea at 20 DAS; remove weeds; ensure good aeration.",
            "kn": "20 DAS ಯೂರಿಯಾ ನೀಡಿ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಗಾಳಿ ಸಂಚಾರ ಇರಲಿ."
        },
        "tillering": {
            "en": "Second nitrogen dose; moisture important; check for shoot borer.",
            "kn": "ಎರಡನೇ ನೈಟ್ರೋಜನ್; ತೇವ ಮುಖ್ಯ; ಶೂಟ್ ಬೋರರ್ ತಡೆ."
        },
        "panicle": {
            "en": "Potash important for grain fill; avoid drought.",
            "kn": "ಪೊಟಾಶ್ ಧಾನ್ಯ ತುಂಬಿಕೆಗೆ ಮುಖ್ಯ; ಬರ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Reduce water; grains harden.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಧಾನ್ಯ ಗಟ್ಟಿಯಾಗುತ್ತದೆ."
        },
        "harvest": {
            "en": "Harvest when earheads turn golden.",
            "kn": "ಮುಳ್ಳುಗಳು ಬಂಗಾರದ ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
    "wheat": {
        "nursery": {
            "en": "Use certified seeds; treat seeds; sow in lines for easy management.",
            "kn": "ಪ್ರಮಾಣಿತ ಬೀಜ ಬಳಸಿ; ಬೀಜ ಶೋಧಿಸಿ; ಸಾಲು ಬಿತ್ತನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "First irrigation at CRI stage; apply first nitrogen split.",
            "kn": "CRI ಹಂತದಲ್ಲಿ ಮೊದಲ ನೀರಾವರಿ; ಮೊದಲ ನೈಟ್ರೋಜನ್ ಹಂಚಿಕೆ ಮಾಡಿ."
        },
        "tillering": {
            "en": "Ensure moisture; remove weeds; apply second nitrogen dose.",
            "kn": "ತೇವ ಇರಲಿ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಎರಡನೇ ನೈಟ್ರೋಜನ್ ನೀಡಿ."
        },
        "panicle": {
            "en": "Critical flowering stage; avoid heat stress; irrigate lightly.",
            "kn": "ಮುಖ್ಯ ಹೂ ಹಂತ; ಉಷ್ಣ ಒತ್ತಡ ತಪ್ಪಿಸಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Stop irrigation; field must dry; grains turn hard.",
            "kn": "ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ; ಹೊಲ ಒಣವಾಗಿರಲಿ; ಧಾನ್ಯ ಗಟ್ಟಿಯಾಗುತ್ತದೆ."
        },
        "harvest": {
            "en": "Harvest when grains turn golden-yellow and firm.",
            "kn": "ಧಾನ್ಯ ಬಂಗಾರದ ಹಳದಿ ಆಗಿ ಗಟ್ಟಿ ಆದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
        "pigeonpea": {
        "nursery": {
            "en": "Direct sowing recommended; treat seeds with Rhizobium + Trichoderma; maintain row spacing 60–90 cm.",
            "kn": "ನೆರಳಾಗಿ ಬಿತ್ತನೆ ಉತ್ತಮ; Rhizobium + Trichoderma ನಿಂದ ಬೀಜ ಶೋಧಿಸಿ; 60–90 ಸೆಂ.ಮೀ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "vegetative": {
            "en": "Apply first split nitrogen and phosphorus; keep field weed-free; perform one intercultivation.",
            "kn": "ಮೊದಲ ನೈಟ್ರೋಜನ್ ಮತ್ತು ಫಾಸ್ಫರಸ್ ನೀಡಿ; ಹೊಲ ಕಳೆ ರಹಿತವಾಗಿಡಿ; ಮಧ್ಯಂತರ ಕಳಿತೋಡು ಮಾಡಿ."
        },
        "branching": {
            "en": "Encourage branching through light topping; ensure adequate sunlight; avoid waterlogging.",
            "kn": "ಲಘು ಟಾಪಿಂಗ್ ಮೂಲಕ ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತೇಜಿಸಿ; ಸಾಕಷ್ಟು ಬೆಳಕು ಇರಲಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Critical stage; avoid moisture stress; apply micronutrients (Boron + Zinc); monitor for pod borer.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ನೀರಿನ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಸೂಕ್ಷ್ಮಾಂಶ (ಬೋರಾನ್ + ಜಿಂಕ್) ನೀಡಿ; ಪಾಡ್ ಬೋರರ್ ತಡೆ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."
        },
        "pod_formation": {
            "en": "Ensure soil moisture; avoid heavy nitrogen; spray neem or biological control for pod borer.",
            "kn": "ಮಣ್ಣಿನ ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ; ಪಾಡ್ ಬೋರರ್‌ಗೆ ನೀಮ್ ಅಥವಾ ಜೈವಿಕ ನಿಯಂತ್ರಣ ಬಳಸಿ."
        },
        "maturity": {
            "en": "Leaves begin to yellow and fall; stop irrigation; prevent lodging.",
            "kn": "ಎಲೆಗಳು ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಬಿದ್ದುಹೋಗುತ್ತವೆ; ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ; ಲಾಡ್ಜಿಂಗ್ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when pods turn brown and dry; thresh carefully to avoid seed damage.",
            "kn": "ಪಾಡ್‌ಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಒಣಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಬೀಜ ಹಾನಿ ತಪ್ಪಿಸಲು ಎಚ್ಚರಿಕೆ ವಹಿಸಿ."
        }
    },
    "moong": {
        "nursery": {
            "en": "Use 15–20 kg seed/ha; treat seeds with Rhizobium; ensure fine tilth soil.",
            "kn": "15–20 ಕೆಜಿ ಬೀಜ/ಹೆ; Rhizobium ನಿಂದ ಬೀಜ ಶೋಧಿಸಿ; ಸೊಂಪಾದ ಮಣ್ಣು ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply light nitrogen; remove weeds; ensure proper aeration.",
            "kn": "ಲಘು ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಗಾಳಿ ಸಂಚಾರ ಇರಲಿ."
        },
        "branching": {
            "en": "Encourage branching by maintaining proper spacing; avoid waterlogging.",
            "kn": "ಸರಿಯಾದ ಅಂತರದಿಂದ ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತಮಗೊಳ್ಳುತ್ತದೆ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Very sensitive stage; avoid moisture stress; apply foliar spray of micronutrients.",
            "kn": "ಅತ್ಯಂತ ಸಂವೇದನಾಶೀಲ ಹಂತ; ನೀರಿನ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಸೂಕ್ಷ್ಮಾಂಶ ಫೋಲಿಯರ್ ಸಿಂಪಡಣೆ ಮಾಡಿ."
        },
        "pod_formation": {
            "en": "Provide adequate moisture; control sucking pests; avoid heavy fertilizer.",
            "kn": "ಸಾಕಷ್ಟು ತೇವ ಇರಲಿ; ಸಕ್ಕಿಂಗ್ ಕೀಟ ನಿಯಂತ್ರಿಸಿ; ಹೆಚ್ಚು ಗೊಬ್ಬರ ಬೇಡ."
        },
        "maturity": {
            "en": "Pods turn black/brown; dry the crop for uniform harvesting.",
            "kn": "ಪಾಡ್‌ಗಳು ಕಪ್ಪು/ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ; ಸಮಾನ ಕೊಯ್ತಿಗಾಗಿ ಒಣಗಲು ಬಿಡಿ."
        },
        "harvest": {
            "en": "Harvest by cutting plants at ground level; thresh after drying.",
            "kn": "ಗಿಡಗಳನ್ನು ನೆಲದ ಮಟ್ಟದಲ್ಲಿ ಕತ್ತರಿಸಿ ಕೊಯ್ಯಿರಿ; ಒಣಗಿಸಿ ತ್ರೇಷರ್‌ನಲ್ಲಿ ತುಪ್ಪಳಿಸಿ."
        }
    },
    "urad": {
        "nursery": {
            "en": "Use 10–15 kg seed/ha; treat seeds with Rhizobium + PSB; sow on well-drained soil.",
            "kn": "10–15 ಕೆಜಿ ಬೀಜ/ಹೆ; Rhizobium + PSB ನಿಂದ ಬೀಜ ಶೋಧಿಸಿ; ಚೆನ್ನಾಗಿ ನೀರು ಹೋದ ಮಣ್ಣಿನಲ್ಲಿ ಬಿತ್ತನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply first nitrogen dose; do early weeding; avoid standing water.",
            "kn": "ಮೊದಲ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮುಂಚಿತ ಕಳೆ ತೆಗೆದುಹಾಕಿ; ನಿಂತ ನೀರು ಬೇಡ."
        },
        "branching": {
            "en": "Good branching improves yield; apply foliar spray of nutrients.",
            "kn": "ಉತ್ತಮ ಶಾಖೆ ಬೆಳವಣಿಗೆ ಫಲವನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ; ಫೋಲಿಯರ್ ಪೋಷಕಾಂಶ ಸಿಂಪಡಣೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Avoid drought; protect from powdery mildew and sucking pests.",
            "kn": "ಬರ ತಪ್ಪಿಸಿ; ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ಮತ್ತು ಸಕ್ಕಿಂಗ್ ಕೀಟಗಳಿಂದ ರಕ್ಷಿಸಿ."
        },
        "pod_formation": {
            "en": "Maintain moisture; avoid overwatering; control pod borer.",
            "kn": "ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ನೀರಾವರಿ ಬೇಡ; ಪಾಡ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Pods dry and turn black; leaves shed naturally.",
            "kn": "ಪಾಡ್‌ಗಳು ಒಣಗಿ ಕಪ್ಪಾಗುತ್ತವೆ; ಎಲೆಗಳು ಸಹಜವಾಗಿ ಬಿದ್ದುಹೋಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest when 80% pods mature; thresh after drying.",
            "kn": "80% ಪಾಡ್‌ಗಳು ಬೆಳೆದಾಗ ಕೊಯ್ಯಿರಿ; ಒಣಗಿಸಿ ತುಪ್ಪಳಿಸಿ."
        }
    },
    "chickpea": {
        "nursery": {
            "en": "Use 80–100 kg seed/ha; treat seeds with Rhizobium + fungicide; sow after first rains.",
            "kn": "80–100 ಕೆಜಿ ಬೀಜ/ಹೆ; Rhizobium + ಫಂಗಿಸೈಡ್ ನಿಂದ ಬೀಜ ಶೋಧಿಸಿ; ಮೊದಲ ಮಳೆ ನಂತರ ಬಿತ್ತನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply basal phosphorus; keep soil slightly moist; control weeds early.",
            "kn": "ಮೂಲ ಫಾಸ್ಫರಸ್ ನೀಡಿ; ಮಣ್ಣು ಸ್ವಲ್ಪ ತೇವ ಇರಲಿ; ಆರಂಭದಲ್ಲೇ ಕಳೆ ನಿಯಂತ್ರಿಸಿ."
        },
        "branching": {
            "en": "Light irrigation to encourage branching; avoid excessive nitrogen.",
            "kn": "ಶಾಖೆ ಬೆಳವಣಿಗೆಗೆ ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ."
        },
        "flowering": {
            "en": "Very sensitive to moisture stress; avoid heavy irrigation; monitor for pod borer.",
            "kn": "ತೇವ ಕೊರತೆಗೆ ಅತ್ಯಂತ ಸಂವೇದನಾಶೀಲ; ಹೆಚ್ಚು ನೀರು ಬೇಡ; ಪಾಡ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "pod_formation": {
            "en": "Ensure soil moisture; do not over-irrigate; apply foliar micronutrients.",
            "kn": "ಮಣ್ಣಿನ ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ನೀರು ಬೇಡ; ಸೂಕ್ಷ್ಮಾಂಶ ಸಿಂಪಡಣೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Plants start yellowing; stop irrigation completely.",
            "kn": "ಗಿಡಗಳು ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ; ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when leaves dry and pods turn brown.",
            "kn": "ಎಲೆಗಳು ಒಣಗಿ ಪಾಡ್‌ಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
    "horsegram": {
        "nursery": {
            "en": "Use 30–40 kg seed/ha; dryland sowing preferred; seed treatment essential.",
            "kn": "30–40 ಕೆಜಿ ಬೀಜ/ಹೆ; ಬರ ಪ್ರದೇಶಕ್ಕೆ ಸೂಕ್ತ; ಬೀಜ ಶೋಧನೆ ಅಗತ್ಯ."
        },
        "vegetative": {
            "en": "Minimal fertilizer needed; weed once at 20–25 DAS.",
            "kn": "ಕಡಿಮೆ ಗೊಬ್ಬರ ಸಾಕು; 20–25 DAS ಕಳೆ ತೆಗೆದುಹಾಕಿ."
        },
        "branching": {
            "en": "Encourage branching through proper spacing; avoid waterlogging.",
            "kn": "ಸರಿಯಾದ ಅಂತರದಿಂದ ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತಮ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Avoid moisture stress; light irrigation if necessary.",
            "kn": "ನೀರಿನ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಅಗತ್ಯವಿದ್ದರೆ ಲಘು ನೀರಾವರಿ."
        },
        "pod_formation": {
            "en": "Maintain light soil moisture; avoid heavy water.",
            "kn": "ಲಘು ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ನೀರು ಬೇಡ."
        },
        "maturity": {
            "en": "Plants dry naturally; pods harden.",
            "kn": "ಗಿಡಗಳು ಸಹಜವಾಗಿ ಒಣಗುತ್ತವೆ; ಪಾಡ್‌ಗಳು ಗಟ್ಟಿಯಾಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Cut plants and dry before threshing.",
            "kn": "ಗಿಡಗಳನ್ನು ಕತ್ತರಿಸಿ ಒಣಗಿಸಿ ನಂತರ ತುಪ್ಪಳಿಸಿ."
        }
    },
    "cowpea": {
        "nursery": {
            "en": "Use 20–25 kg seed/ha; treat seeds with Rhizobium; ensure well-drained soil.",
            "kn": "20–25 ಕೆಜಿ ಬೀಜ/ಹೆ; Rhizobium ನಿಂದ ಬೀಜ ಶೋಧಿಸಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply first dose nitrogen; remove weeds; ensure good sunlight.",
            "kn": "ಮೊದಲ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಬೆಳಕು ಸಾಕಷ್ಟು."
        },
        "branching": {
            "en": "Encourage branching; keep field weed-free; avoid excessive irrigation.",
            "kn": "ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತೇಜಿಸಿ; ಕಳೆ ಇಲ್ಲದಂತೆ ಇರಲಿ; ಹೆಚ್ಚು ನೀರು ಬೇಡ."
        },
        "flowering": {
            "en": "Avoid moisture stress; monitor for aphids and whiteflies.",
            "kn": "ನೀರಿನ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಆಫಿಡ್ ಮತ್ತು ವೈಟ್‌ಫ್ಲೈ ಕೀಟ ತಡೆ ಮಾಡಿ."
        },
        "pod_formation": {
            "en": "Maintain uniform moisture; apply micronutrient spray.",
            "kn": "ಸಮ ತೇವ ಇರಲಿ; ಸೂಕ್ಷ್ಮಾಂಶ ಸಿಂಪಡಣೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Pods turn brown; dry weather important.",
            "kn": "ಪಾಡ್‌ಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ; ಒಣ ವಾತಾವರಣ ಮುಖ್ಯ."
        },
        "harvest": {
            "en": "Harvest at full pod maturity; dry and thresh.",
            "kn": "ಪೂರ್ಣ ಬೆಳೆಯಾದ ಪಾಡ್‌ಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಒಣಗಿಸಿ ತುಪ್ಪಳಿಸಿ."
        }
    },

    "groundnut": {
        "nursery": {
            "en": "Use bold, disease-free seeds; treat with Rhizobium + Trichoderma; sow at 30–45 cm spacing; ensure good soil tilth.",
            "kn": "ದಪ್ಪ, ರೋಗರಹಿತ ಬೀಜ ಬಳಸಿ; Rhizobium + Trichoderma ಶೋಧನೆ ಮಾಡಿ; 30–45 ಸೆಂ.ಮೀ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ; ಮಣ್ಣು ಸೊಂಪಾಗಿರಲಿ."
        },
        "vegetative": {
            "en": "Apply gypsum and first nitrogen dose; remove early weeds; maintain light moisture.",
            "kn": "ಜಿಪ್ಸಮ್ ಮತ್ತು ಮೊದಲ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಆರಂಭಿಕ ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಲಘು ತೇವ ಇರಲಿ."
        },
        "branching": {
            "en": "Critical pegging stage; avoid disturbance to soil; maintain moisture; no deep cultivation.",
            "kn": "ಪೆಗ್ಗಿಂಗ್ ಹಂತ ಮುಖ್ಯ; ಮಣ್ಣಿಗೆ ಅಡ್ಡಿ ಮಾಡಬೇಡಿ; ತೇವ ಇರಲಿ; ಆಳವಾದ ಕಳಿತೋಡು ಬೇಡ."
        },
        "flowering": {
            "en": "Maintain continuous moisture; apply boron to prevent hollow nuts; avoid waterlogging.",
            "kn": "ನಿರಂತರ ತೇವ ಇರಲಿ; ಬೋರ್ deficiency ತಪ್ಪಿಸಲು ಬೋರಾನ್ ನೀಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "pod_formation": {
            "en": "Light irrigation essential; avoid drought stress; protect from leaf spot and rust.",
            "kn": "ಲಘು ನೀರಾವರಿ ಅಗತ್ಯ; ಬರ ತಪ್ಪಿಸಿ; ಲೀಫ್ ಸ್ಪಾಟ್ ಮತ್ತು ರಸ್ಟ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Leaves turn yellow; pods harden; stop irrigation 10–15 days before harvest.",
            "kn": "ಎಲೆಗಳು ಹಳದಿ ಆಗುತ್ತವೆ; ಪಾಡ್ ಗಟ್ಟಿಯಾಗುತ್ತದೆ; ಕೊಯ್ತಿಗೆ 10–15 ದಿನ ಮೊದಲು ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when 70% pods mature; dry well before storage.",
            "kn": "70% ಪಾಡ್ ಬೆಳೆದಾಗ ಕೊಯ್ಯಿರಿ; ಸಂಗ್ರಹಣೆಗೆ ಮುಂಚೆ ಚೆನ್ನಾಗಿ ಒಣಗಿಸಿ."
        }
    },
    "sunflower": {
        "nursery": {
            "en": "Use 8–10 kg seed/ha; ensure good drainage; treat seeds with imidacloprid/fungicide.",
            "kn": "8–10 ಕೆಜಿ ಬೀಜ/ಹೆ; ಚೆನ್ನಾಗಿ ನೀರು ಹೋದ ಮಣ್ಣು ಇರಲಿ; ಬೀಜಗಳನ್ನು ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್/ಫಂಗಿಸೈಡ್ ಶೋಧನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply basal fertilizers; maintain optimum spacing; remove weeds at 20 DAS.",
            "kn": "ಮೂಲ ಗೊಬ್ಬರ ನೀಡಿ; ಸರಿಯಾದ ಅಂತರ ಇರಲಿ; 20 DAS ಕಳೆ ತೆಗೆದುಹಾಕಿ."
        },
        "branching": {
            "en": "Sunflower is non-branching but stem strengthening essential; ensure sunlight and nitrogen.",
            "kn": "ಸನ್ಫ್ಲೋವರ್ ಸಾಮಾನ್ಯವಾಗಿ ಶಾಖೆ ಬಿಡುವುದಿಲ್ಲ; ಆದರೆ ತೊಗಲು ಬಲ ಹೆಚ್ಚಿಸಬೇಕು; ಬೆಳಕು ಮತ್ತು ನೈಟ್ರೋಜನ್ ಇರಲಿ."
        },
        "flowering": {
            "en": "Critical stage; moisture essential; avoid high nitrogen; protect from capitulum borer.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ತೇವ ಅಗತ್ಯ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ; ಕ್ಯಾಪಿಟುಲಮ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "seed_formation": {
            "en": "Ensure potash supply; maintain moisture; avoid waterlogging.",
            "kn": "ಪೊಟಾಶ್ ಪೂರೈಕೆ ಇರಲಿ; ತೇವ ಇರಲಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Back of head turns lemon yellow; petals dry.",
            "kn": "ಹೆಡ್ ಹಿಂಭಾಗ ಲೆಮನ್ ಹಳದಿ ಆಗುತ್ತದೆ; ಹೂ ಕಿರೀಟ ಒಣಗುತ್ತದೆ."
        },
        "harvest": {
            "en": "Harvest when head turns yellow-brown; dry seeds before storage.",
            "kn": "ಹೆಡ್ ಹಳದಿ-ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಬೀಜಗಳನ್ನು ಒಣಗಿಸಿ ಸಂಗ್ರಹಿಸಿ."
        }
    },
      "soybean": {
        "nursery": {
            "en": "Use certified seeds; treat with Rhizobium + PSB; sow at 30–45 cm spacing.",
            "kn": "ಪ್ರಮಾಣಿತ ಬೀಜ ಬಳಸಿ; Rhizobium + PSB ಶೋಧನೆ ಮಾಡಿ; 30–45 ಸೆಂ.ಮೀ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "vegetative": {
            "en": "Apply first nitrogen split; maintain aerobic soil; early weed removal essential.",
            "kn": "ಮೊದಲ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಣ್ಣು ಗಾಳಿಯುಳ್ಳಂತಿರಲಿ; ಆರಂಭಿಕ ಕಳೆ ತೆಗೆದುಹಾಕಿ."
        },
        "branching": {
            "en": "Maintain adequate sunlight; apply micronutrients for branching.",
            "kn": "ಸಾಕಷ್ಟು ಬೆಳಕು ಇರಲಿ; ಶಾಖೆ ಬೆಳವಣಿಗೆಗೆ ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ."
        },
        "flowering": {
            "en": "Very sensitive to moisture stress; avoid drought; protect from rust and stem fly.",
            "kn": "ನೀರಿನ ಕೊರತೆಗೆ ತುಂಬಾ ಸಂವೇದನಾಶೀಲ; ಬರ ತಪ್ಪಿಸಿ; ರಸ್ಟ್ ಮತ್ತು ಸ್ಟೆಮ್ ಫ್ಲೈ ತಡೆ ಮಾಡಿ."
        },
        "pod_formation": {
            "en": "Maintain uniform moisture; apply potash; avoid waterlogging.",
            "kn": "ಸಮ ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ಪೂರೈಕೆ ಇರಲಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Leaves yellow and fall; pods harden.",
            "kn": "ಎಲೆಗಳು ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಬಿದ್ದುಹೋಗುತ್ತವೆ; ಪಾಡ್ ಗಟ್ಟಿ ಆಗುತ್ತದೆ."
        },
        "harvest": {
            "en": "Harvest when pods turn brown and dry; thresh carefully.",
            "kn": "ಪಾಡ್‌ಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಒಣಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಎಚ್ಚರಿಕೆಯಿಂದ ತುಪ್ಪಳಿಸಿ."
        }
    },

    "sesame": {
        "nursery": {
            "en": "Use 3–4 kg seed/ha; treat seeds; ensure fine tilth soil; sow shallow.",
            "kn": "3–4 ಕೆಜಿ ಬೀಜ/ಹೆ; ಬೀಜ ಶೋಧಿಸಿ; ಸೊಂಪಾದ ಮಣ್ಣು ಇರಲಿ; ಮೇಲ್ಮಟ್ಟದಲ್ಲಿ ಬಿತ್ತನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; remove weeds early; avoid waterlogging.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಆರಂಭಿಕ ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "branching": {
            "en": "Encourage branching by proper spacing; avoid drought stress.",
            "kn": "ಸರಿಯಾದ ಅಂತರದಿಂದ ಶಾಖೆ ಬೆಳವಣಿಗೆ ಸುಧಾರಿಸುತ್ತದೆ; ಬರ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Moisture critical; apply micronutrients; protect from leaf spot.",
            "kn": "ತೇವ ಮುಖ್ಯ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಲೀಫ್ ಸ್ಪಾಟ್ ತಡೆ ಮಾಡಿ."
        },
        "seed_formation": {
            "en": "Maintain moisture; ensure potash availability.",
            "kn": "ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ಪೂರೈಕೆ ಇರಲಿ."
        },
        "maturity": {
            "en": "Capsules turn yellow-brown; leaves shed naturally.",
            "kn": "ಕ್ಯಾಪ್ಸೂಲ್‌ಗಳು ಹಳದಿ-ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ; ಎಲೆಗಳು ಸಹಜವಾಗಿ ಬಿದ್ದುಹೋಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest before capsule shattering; dry well.",
            "kn": "ಕ್ಯಾಪ್ಸೂಲ್ ಒಡೆಹೋದಕ್ಕೆ ಮುನ್ನ ಕೊಯ್ಯಿರಿ; ಚೆನ್ನಾಗಿ ಒಣಗಿಸಿ."
        }
    },
    "castor": {
        "nursery": {
            "en": "Use 8–10 kg seed/ha; treat seeds with fungicide; sow in well-drained soil.",
            "kn": "8–10 ಕೆಜಿ ಬೀಜ/ಹೆ; ಫಂಗಿಸೈಡ್ ಶೋಧನೆ ಮಾಡಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣಿನಲ್ಲಿ ಬಿತ್ತನೆ."
        },
        "vegetative": {
            "en": "Apply nitrogen; remove weeds; maintain good spacing.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಸರಿಯಾದ ಅಂತರ ಇರಲಿ."
        },
        "branching": {
            "en": "Encourage branching; apply micronutrients; avoid waterlogging.",
            "kn": "ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತೇಜಿಸಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Maintain moisture; protect from capsule borer; avoid excess nitrogen.",
            "kn": "ತೇವ ಇರಲಿ; ಕ್ಯಾಪ್ಸೂಲ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ."
        },
        "seed_formation": {
            "en": "Ensure potash supply; maintain uniform moisture.",
            "kn": "ಪೊಟಾಶ್ ಪೂರೈಕೆ ಇರಲಿ; ಸಮ ತೇವ ಇರಲಿ."
        },
        "maturity": {
            "en": "Capsules dry and harden; leaves turn yellow.",
            "kn": "ಕ್ಯಾಪ್ಸೂಲ್‌ಗಳು ಒಣಗಿ ಗಟ್ಟಿಯಾಗುತ್ತವೆ; ಎಲೆಗಳು ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest when capsules turn brown and dry.",
            "kn": "ಕ್ಯಾಪ್ಸೂಲ್‌ಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಒಣಗಿದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
    "safflower": {
        "nursery": {
            "en": "Use 8–10 kg seed/ha; sow on ridges for better drainage; treat seeds with fungicide.",
            "kn": "8–10 ಕೆಜಿ ಬೀಜ/ಹೆ; ನೀರು ಹೋದಂತೆ ರಿಡ್ಜ್ ಮೇಲೆ ಬಿತ್ತನೆ ಮಾಡಿ; ಫಂಗಿಸೈಡ್ ಶೋಧನೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; remove weeds early; maintain low moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಆರಂಭಿಕ ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಕಡಿಮೆ ತೇವ ಉಳಿಸಿ."
        },
        "branching": {
            "en": "Encourage branching; avoid excess irrigation.",
            "kn": "ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತೇಜಿಸಿ; ಹೆಚ್ಚು ನೀರಾವರಿ ಬೇಡ."
        },
        "flowering": {
            "en": "Moisture important; protect from aphids and powdery mildew.",
            "kn": "ತೇವ ಮುಖ್ಯ; ಆಫಿಡ್ ಮತ್ತು ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ತಡೆ ಮಾಡಿ."
        },
        "seed_formation": {
            "en": "Ensure potash availability; keep field aerated.",
            "kn": "ಪೊಟಾಶ್ ಪೂರೈಕೆ ಇರಲಿ; ಹೊಲ ಗಾಳಿಯುಳ್ಳಂತಿರಲಿ."
        },
        "maturity": {
            "en": "Flower heads dry and harden; leaves dry.",
            "kn": "ಹೂಮುಳ್ಳುಗಳು ಒಣಗಿ ಗಟ್ಟಿಯಾಗುತ್ತವೆ; ಎಲೆಗಳು ಒಣಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest when flower heads turn brown and dry.",
            "kn": "ಮುಳ್ಳುಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಒಣಗಿದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
    "sugarcane": {
        "planting": {
            "en": "Use healthy sett pieces with 2–3 buds; treat setts with fungicide; plant in well-prepared furrows 1.2–1.5 m apart.",
            "kn": "2–3 ಮೊಗ್ಗುಗಳಿರುವ ಆರೋಗ್ಯಕರ ಸೆಟ್‌ಗಳನ್ನು ಬಳಸಿ; ಫಂಗಿಸೈಡ್ ಶೋಧನೆ ಮಾಡಿ; 1.2–1.5 ಮೀ ಅಂತರದ ಫರೋಗಳಲ್ಲಿ ಬಿತ್ತನೆ ಮಾಡಿ."
        },
        "germination": {
            "en": "Ensure light irrigation; protect buds from termites; apply basal fertilizers including FYM.",
            "kn": "ಲಘು ನೀರಾವರಿ ನೀಡಿ; ಟರ್ಮೈಟ್‌ನಿಂದ ಮೊಗ್ಗುಗಳನ್ನು ರಕ್ಷಿಸಿ; FYM ಸೇರಿದಂತೆ ಮೂಲ ಗೊಬ್ಬರ ನೀಡಿ."
        },
        "early_vegetative": {
            "en": "Start earthing-up; maintain moisture; apply nitrogen split; control weeds.",
            "kn": "ಇರ್ಥಿಂಗ್-ಅಪ್ ಪ್ರಾರಂಭಿಸಿ; ತೇವ ಇರಲಿ; ನೈಟ್ರೋಜನ್ ಹಂತವಾಗಿ ನೀಡಿ; ಕಳೆ ನಿಯಂತ್ರಿಸಿ."
        },
        "vegetative": {
            "en": "Rapid growth stage; irrigate regularly; apply second split nitrogen; maintain trash mulching.",
            "kn": "ವೇಗವಾಗಿ ಬೆಳೆಯುವ ಹಂತ; ನಿಯಮಿತ ನೀರಾವರಿ ಮಾಡಿ; ಎರಡನೇ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಟ್ರಾಶ್ ಮಲ್ಚಿಂಗ್ ಮಾಡಿ."
        },
        "grand_growth": {
            "en": "Critical stage for yield; ensure continuous moisture; apply potash; protect from borers.",
            "kn": "ಉತ್ಪಾದನೆಗೆ ಪ್ರಮುಖ ಹಂತ; ನಿರಂತರ ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಬೋರರ್ ಕೀಟದಿಂದ ರಕ್ಷಿಸಿ."
        },
        "maturity": {
            "en": "Reduce irrigation; internodes harden; stop nitrogen completely.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಇಂಟರ್‌ನೋಡ್ಗಳು ಗಟ್ಟಿಯಾಗುತ್ತವೆ; ನೈಟ್ರೋಜನ್ ನಿಲ್ಲಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when cane juice turns sweet and brix increases; avoid burning cane.",
            "kn": "ಕಬ್ಬು ರಸ ಸಿಹಿಯಾಗುತ್ತಾ ಬೃಹತ್ ಹೆಚ್ಚಾದಾಗ ಕೊಯ್ಯಿರಿ; ಕಬ್ಬು ಸುಡುವುದನ್ನು ತಪ್ಪಿಸಿ."
        },
        "ratoon_management": {
            "en": "After harvest, cut stalks at ground level; apply nitrogen and irrigate for ratoon crop.",
            "kn": "ಕೊಯ್ತಿನ ನಂತರ ನೆಲದ ಮಟ್ಟದಲ್ಲಿ ಕತ್ತರಿಸಿ; ನೈಟ್ರೋಜನ್ ನೀಡಿ ಹಾಗೂ ರಟೂನ್ ಬೆಳೆಗಾಗಿ ನೀರಾವರಿ ಮಾಡಿ."
        }
    },
    "cotton": {
        "nursery": {
            "en": "Direct sowing preferred; treat seeds with fungicide and imidacloprid; ensure good soil moisture at sowing.",
            "kn": "ನೆರಳಾಗಿ ಬಿತ್ತನೆ ಉತ್ತಮ; ಬೀಜಗಳನ್ನು ಫಂಗಿಸೈಡ್ ಮತ್ತು ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ನಿಂದ ಶೋಧಿಸಿ; ಬಿತ್ತನೆಗೆ ಮಣ್ಣು ತೇವ ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply basal fertilizers; first weeding at 20–25 DAS; monitor for sucking pests.",
            "kn": "ಮೂಲ ಗೊಬ್ಬರ ನೀಡಿ; 20–25 DAS ಕಳೆ ತೆಗೆದುಹಾಕಿ; ಸಕ್ಕಿಂಗ್ ಕೀಟಗಳನ್ನು ತಡೆಯಿರಿ."
        },
        "square_formation": {
            "en": "Important stage; apply nitrogen and potash; avoid moisture stress; monitor for leaf curling virus.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ನೈಟ್ರೋಜನ್ ಮತ್ತು ಪೊಟಾಶ್ ನೀಡಿ; ತೇವ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಎಲೆ ಕರ್ಭಟದ ವೈರಸ್ ತಡೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Irrigation essential; apply micronutrients; protect from bollworms; avoid excess nitrogen.",
            "kn": "ನೀರಾವರಿ ಅಗತ್ಯ; ಸೂಕ್ಷ್ಮಾಂಶ ಸಿಂಪಡಣೆ ಮಾಡಿ; ಬಾಲ್ವೋರ್ಮ್ ತಡೆ ಮಾಡಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ."
        },
        "boll_formation": {
            "en": "Maintain uniform moisture; apply potash; avoid waterlogging; protect from pink bollworm.",
            "kn": "ಸಮ ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ; ಪಿಂಕ್ ಬಾಲ್ವೋರ್ಮ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Leaves start drying; stop irrigation; ensure boll opening.",
            "kn": "ಎಲೆಗಳು ಒಣಗಲು ಪ್ರಾರಂಭ; ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ; ಬೋಲ್ ತೆರೆದುಕೊಳ್ಳಲು ಸಹಕಾರ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest clean, white, fully opened bolls; avoid contamination.",
            "kn": "ಸ್ವಚ್ಛ, ಬಿಳಿ, ಸಂಪೂರ್ಣ ತೆರೆದ ಬೋಲ್‌ಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಮಾಲಿನ್ಯ ತಪ್ಪಿಸಿ."
        }
    },
    "tobacco": {
        "nursery": {
            "en": "Prepare raised nursery beds; treat seeds; maintain light irrigation; protect seedlings from damping-off.",
            "kn": "ಎತ್ತಿದ ನರ್ಸರಿ ಬೆಡ್ ತಯಾರಿಸಿ; ಬೀಜ ಶೋಧನೆ ಮಾಡಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಡ್ಯಾಂಪಿಂಗ್-ಆಫ್ ತಡೆ ಮಾಡಿ."
        },
        "transplanting": {
            "en": "Transplant 30–35 day healthy seedlings; avoid deep planting; irrigate lightly.",
            "kn": "30–35 ದಿನದ ಆರೋಗ್ಯಕರ ಮೊಳಕೆಗಳನ್ನು ನಾಟಿಕೆ ಮಾಡಿ; ಆಳವಾಗಿ ನೆಡಬೇಡಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ."
        },
        "early_vegetative": {
            "en": "Apply nitrogen; keep field weed-free; avoid excess water.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಹೊಲ ಕಳೆ ರಹಿತವಾಗಿಡಿ; ಹೆಚ್ಚು ನೀರು ಬೇಡ."
        },
        "vegetative": {
            "en": "Maintain moisture; apply potash for leaf quality; monitor for aphids and whitefly.",
            "kn": "ತೇವ ಇರಲಿ; ಎಲೆ ಗುಣಮಟ್ಟಕ್ಕೆ ಪೊಟಾಶ್ ನೀಡಿ; ಆಫಿಡ್ ಮತ್ತು ವೈಟ್‌ಫ್ಲೈ ತಡೆ ಮಾಡಿ."
        },
        "leaf_expansion": {
            "en": "Critical stage for leaf size; avoid stress; apply micronutrients; control leaf spot.",
            "kn": "ಎಲೆ ವಿಸ್ತರಣೆ ಮುಖ್ಯ; ಒತ್ತಡ ತಪ್ಪಿಸಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಲೀಫ್ ಸ್ಪಾಟ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Leaves turn yellow-green; reduce irrigation; prepare for harvesting.",
            "kn": "ಎಲೆಗಳು ಹಳದಿ-ಹಸಿರು ಆಗುತ್ತವೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಕೊಯ್ತಿಗೆ ಸಿದ್ಧತೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest mature leaves in 4–6 pickings; cure leaves immediately.",
            "kn": "4–6 ಹಂತಗಳಲ್ಲಿ ಬೆಳೆದ ಎಲೆಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ತಕ್ಷಣ ಕ್ಯೂರ್ ಮಾಡಿ."
        }
    },

    "coffee": {
        "seedling": {
            "en": "Raise seedlings in shade; ensure well-drained soil; protect from damping-off with fungicide.",
            "kn": "ನೆರಳಿನಲ್ಲಿರುವ ನರ್ಸರಿಯಲ್ಲಿ ಮೊಳಕೆ ಬೆಳೆಸಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಬಳಸಿ; ಡ್ಯಾಂಪಿಂಗ್‌-ಆಫ್ ತಡೆಗೆ ಫಂಗಿಸೈಡ್ ಬಳಸಿ."
        },
        "juvenile": {
            "en": "Provide 50–60% shade; apply FYM; control weeds; irrigate lightly during dry months.",
            "kn": "50–60% ನೆರಳು ಇರಲಿ; FYM ನೀಡಿ; ಕಳೆ ನಿಯಂತ್ರಿಸಿ; ಒಣಗಾಲದಲ್ಲಿ ಲಘು ನೀರಾವರಿ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Prune suckers; maintain balanced shade; apply NPK and micronutrients; manage white stem borer.",
            "kn": "ಸಕ್ಕರ್‌ಗಳನ್ನು ತೆಗೆಯಿರಿ; ಸಮತಟ್ಟಾದ ನೆರಳು ಇರಲಿ; NPK ಮತ್ತು ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ವೈಟ್ ಸ್ಟೆಮ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Blossom irrigation essential; provide backing irrigation after 7–10 days; avoid stress.",
            "kn": "ಬ್ಲಾಸಮ್ ನೀರಾವರಿ ಅಗತ್ಯ; 7–10 ದಿನಗಳಲ್ಲಿ ಬ್ಯಾಕಿಂಗ್ ನೀರಾವರಿ ನೀಡಿ; ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "fruiting": {
            "en": "Ensure moisture; apply potash for berry development; control berry borer.",
            "kn": "ತೇವ ಇರಲಿ; ಬೆರ್ರಿ ಬೆಳವಣಿಗೆಗೆ ಪೊಟಾಶ್ ನೀಡಿ; ಬೆರ್ರಿ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Berries turn red; reduce irrigation; prevent fungal diseases.",
            "kn": "ಬೆರ್ರಿಗಳು ಕೆಂಪಾಗುವ ಹಂತ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಫಂಗಸ್ ರೋಗ ತಡೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Pick only red, ripe berries; avoid stripping; dry properly for quality.",
            "kn": "ಪೂರ್ಣ ಬೆಳೆದ ಕೆಂಪು ಬೆರ್ರಿಗಳನ್ನು ಮಾತ್ರ ಕೊಯ್ಯಿರಿ; ಸ್ಟ್ರಿಪ್ಪಿಂಗ್ ಬೇಡ; ಗುಣಮಟ್ಟಕ್ಕಾಗಿ ಸರಿಯಾಗಿ ಒಣಗಿಸಿ."
        },
        "rejuvenation": {
            "en": "Post-harvest pruning; manure application; maintain shade trees.",
            "kn": "ಕೊಯ್ತಿನ ನಂತರ ಕತ್ತರಿಸಿರಿ; ಗೊಬ್ಬರ ನೀಡಿ; ನೆರಳು ಮರಗಳನ್ನು ನಿರ್ವಹಿಸಿ."
        }
    },
    "tea": {
        "seedling": {
            "en": "Raise seedlings under shade nets; use well-drained acidic soil; protect from pests.",
            "kn": "ನೆರಳು ನೆಟ್ಅಡಿಯಲ್ಲಿ ನರ್ಸರಿ ಬೆಳೆಸಿರಿ; ಆಮ್ಲೀಯ ಮತ್ತು ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಬಳಸಿ; ಕೀಟಗಳಿಂದ ರಕ್ಷಿಸಿ."
        },
        "juvenile": {
            "en": "Light pruning to encourage branching; maintain shade; irrigate regularly.",
            "kn": "ಶಾಖೆ ಬೆಳವಣಿಗೆಗೆ ಲಘು ಪ್ರೂನಿಂಗ್ ಮಾಡಿ; ನೆರಳು ಇರಲಿ; ನೀರಾವರಿ ನಿಯಮಿತವಾಗಿ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen frequently; pluck at 7–10 day intervals; manage mites and looper pests.",
            "kn": "ನೈಟ್ರೋಜನ್ ನಿಯಮಿತವಾಗಿ ನೀಡಿ; 7–10 ದಿನಗಳಿಗೊಮ್ಮೆ ಪ್ಲಕ್ಕಿಂಗ್ ಮಾಡಿ; ಮೈಟ್ ಮತ್ತು ಲೂಪರ್ ಕೀಟ ತಡೆ ಮಾಡಿ."
        },
        "flush_growth": {
            "en": "Young shoots develop; ensure moisture; manage shading for quality.",
            "kn": "ಹೊಸ ಮೊಗ್ಗುಗಳು ಬೆಳೆಯುತ್ತವೆ; ತೇವ ಇರಲಿ; ಗುಣಮಟ್ಟಕ್ಕಾಗಿ ನೆರಳು ನಿಯಂತ್ರಿಸಿ."
        },
        "maturity": {
            "en": "Reduce shade; maintain pruning cycles; prepare for harvest season.",
            "kn": "ನೆರಳು ಕಡಿಮೆ ಮಾಡಿ; ಪ್ರೂನಿಂಗ್ ಚಕ್ರಗಳನ್ನು ನಿರ್ವಹಿಸಿ; ಕೊಯ್ತಿಗೆ ಸಿದ್ಧತೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest two leaves and a bud; frequent plucking improves quality.",
            "kn": "ಎರಡು ಎಲೆ + ಒಂದು ಮೊಗ್ಗನ್ನು ಕೊಯ್ಯಿರಿ; ನಿಯಮಿತ ಪ್ಲಕ್ಕಿಂಗ್ ಗುಣಮಟ್ಟ ಹೆಚ್ಚಿಸುತ್ತದೆ."
        },
        "rejuvenation": {
            "en": "Hard pruning every few years to restore vigor.",
            "kn": "ಬೆಳೆ ಶಕ್ತಿ ಮರಳಿ ಪಡೆಯಲು ಕೆಲವು ವರ್ಷಗಳಿಗೊಮ್ಮೆ ಕಠಿಣ ಕತ್ತರಿಸಿರಿ."
        }
    },
    "arecanut": {
        "seedling": {
            "en": "Select healthy seedlings; plant in pits filled with compost and sand; provide temporary shade.",
            "kn": "ಆರೋಗ್ಯಕರ ಮೊಳಕೆ ಆಯ್ಕೆ ಮಾಡಿ; ಕಂಪೋಸ್ಟ್ + ಮರಳು ಮಿಶ್ರಣದ ಗುಂಡಿಯಲ್ಲಿ ನೆಡಿರಿ; ತಾತ್ಕಾಲಿಕ ನೆರಳು ಒದಗಿಸಿ."
        },
        "juvenile": {
            "en": "Irrigate lightly; apply FYM annually; prevent sun scorch; control spindle bugs.",
            "kn": "ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ವರ್ಷಕ್ಕೊಮ್ಮೆ FYM ನೀಡಿ; ಸನ್ ಸ್ಕೋರ್ಚ್ ತಪ್ಪಿಸಿ; ಸ್ಪಿಂಡಲ್ ಬಗ್ ತಡೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Maintain regular irrigation; apply NPK; keep basin weed-free.",
            "kn": "ನಿಯಮಿತ ನೀರಾವರಿ ಮಾಡಿ; NPK ನೀಡಿ; ಬೇಸಿನ್ ಕಳೆ ರಹಿತವಾಗಿಡಿ."
        },
        "flowering": {
            "en": "Maintain moderate moisture; avoid drought; control mite infestations.",
            "kn": "ಮಧ್ಯಮ ತೇವ ಇರಲಿ; ಬರ ತಪ್ಪಿಸಿ; ಮೈಟ್ ಕೀಟ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain irrigation; protect from fruit rot and yellow leaf disease.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ; ಫ್ರೂಟ್ ರಾಟ್ ಮತ್ತು ಯೆಲ್ಲೋ ಲೀಫ್ ರೋಗ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Nuts turn yellow/orange depending on variety; reduce irrigation.",
            "kn": "ಗೊಡಂಬಿ ಹಳದಿ/ಕೆಂಪು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest mature nuts; dry properly before storage.",
            "kn": "ಬೆಳೆದ ಗೊಡಂಬಿ ಕೊಯ್ಯಿರಿ; ಸಂಗ್ರಹಣೆಗೆ ಮುಂಚೆ ಒಣಗಿಸಿ."
        },
        "rejuvenation": {
            "en": "Remove old fronds; apply compost; manage spacing by thinning old palms.",
            "kn": "ಹಳೆಯ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ಹಳೆಯ ಮರಗಳನ್ನು ತೆಳುವಾಗಿಸಿ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        }
    },
    "coconut": {
        "seedling": {
            "en": "Select tall, healthy seedlings; plant in large pits with compost; provide partial shade.",
            "kn": "ಉದ್ದ, ಆರೋಗ್ಯಕರ ಮೊಳಕೆ ಆಯ್ಕೆ ಮಾಡಿ; ದೊಡ್ಡ ಗುಂಡಿಗಳಲ್ಲಿ ಕಂಪೋಸ್ಟ್ ಸೇರಿಸಿ ನೆಡಿರಿ; ಭಾಗಶಃ ನೆರಳು ನೀಡಿ."
        },
        "juvenile": {
            "en": "Irrigate regularly; apply organic manure; protect from rhinoceros beetle.",
            "kn": "ನಿಯಮಿತ ನೀರಾವರಿ ಮಾಡಿ; ಆರ್ಗಾನಿಕ್ ಗೊಬ್ಬರ ನೀಡಿ; ರೈನೋಸೆರೋಸ್ ಬೀಟಲ್ ತಡೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply NPK every 6 months; maintain clean basins; control termite and eriophyid mites.",
            "kn": "ಪ್ರತಿ 6 ತಿಂಗಳಿಗೆ NPK ನೀಡಿ; ಬೇಸಿನ್ ಸ್ವಚ್ಛ ಇರಲಿ; ಟರ್ಮೈಟ್ ಮತ್ತು ಎರಿಯೋಫೈಡ್ ಮೈಟ್ ತಡೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Ensure moisture; apply boron; protect inflorescence from insects.",
            "kn": "ತೇವ ಇರಲಿ; ಬೋರಾನ್ ನೀಡಿ; ಹೂಮಾಲೆಗಳನ್ನು ಕೀಟದಿಂದ ರಕ್ಷಿಸಿ."
        },
        "fruiting": {
            "en": "Critical stage; maintain irrigation; apply potash; control nut fall.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ನೀರಾವರಿ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ನಟ್ ಫಾಲ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Nuts harden; reduce irrigation; prepare for harvest.",
            "kn": "ಗೊಡಂಬಿ ಗಟ್ಟಿಯಾಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಕೊಯ್ತಿಗೆ ಸಿದ್ಧತೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest tender or mature nuts depending on use.",
            "kn": "ಬಳಕೆಗೆ ಅನುಗುಣವಾಗಿ ಮನೆಗೊಡಂಬಿ ಅಥವಾ ಬೆಳೆದ ಗೊಡಂಬಿ ಕೊಯ್ಯಿರಿ."
        },
        "rejuvenation": {
            "en": "Apply compost; remove diseased fronds; clean crown regularly.",
            "kn": "ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ರೋಗಗ್ರಸ್ತ ಎಲೆ ತೆಗೆಯಿರಿ; ಕ್ರೌನ್ ಸ್ವಚ್ಛ ಇಟ್ಟುಕೊಳ್ಳಿ."
        
        }
    },

    "rubber": {
        "seedling": {
            "en": "Raise seedlings in polybags; transplant during early monsoon; maintain shade.",
            "kn": "ಪಾಲಿಬ್ಯಾಗ್‌ಗಳಲ್ಲಿ ನರ್ಸರಿ ಬೆಳೆಸಿರಿ; ಮೊದಲ ಮಳೆಗಾಲದಲ್ಲಿ ನಾಟಿಕೆ ಮಾಡಿ; ನೆರಳು ಇರಲಿ."
        },
        "juvenile": {
            "en": "Apply fertilizers regularly; maintain weed-free basin; avoid waterlogging.",
            "kn": "ನಿಯಮಿತ ಗೊಬ್ಬರ ನೀಡಿ; ಬೇಸಿನ್ ಕಳೆ ರಹಿತವಾಗಿಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "vegetative": {
            "en": "Encourage straight growth; control leaf diseases; maintain moisture.",
            "kn": "ನೆಟ್ಟ ಬೆಳವಣಿಗೆ ಉತ್ತೇಜಿಸಿ; ಎಲೆ ರೋಗ ತಡೆ ಮಾಡಿ; ತೇವ ಇರಲಿ."
        },
        "immature_growth": {
            "en": "Maintain soil fertility; avoid bark injuries; ensure sunlight.",
            "kn": "ಮಣ್ಣಿನ ಫಲವತ್ತತೆ ಉಳಿಸಿ; ತೊಗಲು ಗಾಯ ತಪ್ಪಿಸಿ; ಬೆಳಕು ಸಾಕಷ್ಟು ಇರಲಿ."
        },
        "maturity": {
            "en": "Trees ready for tapping when girth reaches 50 cm; stop irrigation.",
            "kn": "ತೊಗಲು ಗಾತ್ರ 50 ಸೆಂ.ಮೀ ಆದಾಗ ಟ್ಯಾಪಿಂಗ್‌ಗೆ ಸಿದ್ಧ; ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ."
        },
        "harvest": {
            "en": "Tap early morning; maintain correct tapping angle; avoid over-tapping.",
            "kn": "ಬೆಳಿಗ್ಗೆ ಟ್ಯಾಪಿಂಗ್ ಮಾಡಿ; ಸರಿಯಾದ ಕೋನ ಕಾಯ್ದುಕೊಳ್ಳಿ; ಅತಿಯಾದ ಟ್ಯಾಪಿಂಗ್ ಬೇಡ."
        },
        "rejuvenation": {
            "en": "Rest trees periodically; apply manure; treat diseases promptly.",
            "kn": "ಮರಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ವಿಶ್ರಾಂತಿ ನೀಡಿ; ಗೊಬ್ಬರ ನೀಡಿ; ರೋಗಗಳನ್ನು ತಕ್ಷಣ ತಡೆ ಮಾಡಿ."
        }
    },

    "cashew": {
        "seedling": {
            "en": "Plant grafted seedlings; ensure well-drained soil; provide initial shade.",
            "kn": "ಗ್ರಾಫ್ಟ್ ಮಾಡಿದ ಮೊಳಕೆಗಳನ್ನು ನೆಡಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ; ಆರಂಭದಲ್ಲಿ ನೆರಳು ಒದಗಿಸಿ."
        },
        "juvenile": {
            "en": "Train branches; apply FYM; control tea mosquito bug.",
            "kn": "ಶಾಖೆಗಳನ್ನು ಸರಿಯಾಗಿ ರೂಪಿಸಿ; FYM ನೀಡಿ; ಟೀ ಮಾಸ್ಕಿಟೋ ಬಗ್ ತಡೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply NPK; maintain basin cleanliness; prune water shoots.",
            "kn": "NPK ನೀಡಿ; ಬೇಸಿನ್ ಸ್ವಚ್ಛ ಇರಲಿ; ವಾಟರ್ ಶೂಟ್‌ಗಳನ್ನು ಕತ್ತರಿಸಿ."
        },
        "flowering": {
            "en": "Ensure light irrigation; protect from stem borer and thrips.",
            "kn": "ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಸ್ಟೆಮ್ ಬೋರರ್ ಮತ್ತು ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Maintain moisture; apply potash; protect from powdery mildew.",
            "kn": "ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Cashew apples turn red/yellow; nuts detach easily.",
            "kn": "ಕಾಜು ಆಪಲ್ ಕೆಂಪು/ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತದೆ; ಗೊಡಂಬಿ ಸುಲಭವಾಗಿ ಬೇರ್ಪಡುತ್ತದೆ."
        },
        "harvest": {
            "en": "Collect nuts after natural fall; dry well before storage.",
            "kn": "ಸ್ವಾಭಾವಿಕವಾಗಿ ಬಿದ್ದ ನಟ್‌ಗಳನ್ನು ಕಲೆಹಾಕಿ; ಒಣಗಿಸಿ ಸಂಗ್ರಹಿಸಿ."
        },
        "rejuvenation": {
            "en": "Prune old branches; apply manure; manage canopy.",
            "kn": "ಹಳೆಯ ಶಾಖೆಗಳನ್ನು ಕತ್ತರಿಸಿ; ಗೊಬ್ಬರ ನೀಡಿ; ಕ್ಯಾನಪಿ ನಿಯಂತ್ರಿಸಿ."
        }
    },
    "mango": {
        "seedling": {
            "en": "Plant healthy grafts; use well-drained soil; provide staking and light shade initially.",
            "kn": "ಆರೋಗ್ಯಕರ ಗ್ರಾಫ್ಟ್ ನೆಡಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಬಳಸಿ; ಆರಂಭದಲ್ಲಿ ಸ್ಟೇಕಿಂಗ್ ಮತ್ತು ಲಘು ನೆರಳು ಒದಗಿಸಿ."
        },
        "vegetative": {
            "en": "Apply FYM and NPK; prune water shoots; maintain basin weed-free.",
            "kn": "FYM ಮತ್ತು NPK ನೀಡಿ; ವಾಟರ್ ಶೂಟ್‌ಗಳನ್ನು ಕತ್ತರಿಸಿ; ಬೇಸಿನ್ ಸ್ವಚ್ಛ ಇರಲಿ."
        },
        "flowering": {
            "en": "Do not irrigate heavily; apply micronutrients; protect panicles from hoppers and powdery mildew.",
            "kn": "ಹೆಚ್ಚು ನೀರಾವರಿ ಬೇಡ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಹಾಪರ್ ಮತ್ತು ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Maintain light moisture; apply potash; prevent fruit drop with NAA spray.",
            "kn": "ಲಘು ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; NAA ಸಿಂಪಡಣೆ ಮೂಲಕ ಹಣ್ಣು ಬಿದ್ದುಹೋಗುವುದನ್ನು ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Fruits change color slightly; reduce irrigation; protect from fruit flies.",
            "kn": "ಹಣ್ಣುಗಳ ಬಣ್ಣ ಸ್ವಲ್ಪ ಬದಲಾಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಫ್ರೂಟ್ ಫ್ಲೈ ತಡೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest mature but firm fruits; avoid dropping; handle gently.",
            "kn": "ಬೆಳೆದ ಆದರೆ ಗಟ್ಟಿಯಾದ ಹಣ್ಣುಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಕೆಳಗೆ ಬಿದ್ದಂತೆ ಬಿಡಬೇಡಿ; ಎಚ್ಚರಿಕೆಯಿಂದ ಹಿಡಿಯಿರಿ."
        },
        "rejuvenation": {
            "en": "Prune after harvest; apply compost; remove diseased branches.",
            "kn": "ಕೊಯ್ತಿನ ನಂತರ ಕತ್ತರಿಸಿ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ರೋಗಗ್ರಸ್ತ ಶಾಖೆ ತೆಗೆಯಿರಿ."
        }
    },  

    "banana": {
        "seedling": {
            "en": "Plant healthy suckers/tissue culture plants; ensure good drainage; apply FYM in pits.",
            "kn": "ಆರೋಗ್ಯಕರ ಸಕ್ಕರ್/ಟಿಸ್ಯೂ ಕಲ್ಚರ್ ಗಿಡಗಳನ್ನು ನೆಡಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ; ಗುಂಡಿಗಳಲ್ಲಿ FYM ನೀಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; desucker every 45 days; maintain irrigation regularly.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಪ್ರತಿ 45 ದಿನಕ್ಕೆ ಸಕ್ಕರ್ ತೆಗೆದುಹಾಕಿ; ನೀರಾವರಿ ನಿಯಮಿತವಾಗಿರಲಿ."
        },
        "flowering": {
            "en": "Support the bunch; apply potash; prevent thrips and weevils.",
            "kn": "ಕಾಯಿ ಮೊಗ್ಗಿಗೆ ಬೆಂಬಲ ನೀಡಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಥ್ರಿಪ್ಸ್ ಮತ್ತು ವೀವಿಲ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Ensure irrigation; apply micronutrients; bag bunches for quality.",
            "kn": "ನೀರಾವರಿ ಇರಲಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಗುಣಮಟ್ಟಕ್ಕಾಗಿ ಬಂಚ್ ಬ್ಯಾಗ್ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Reduce irrigation; fruits turn dark green or pale yellow depending on variety.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಹಣ್ಣುಗಳು ಕತ್ತಲೆ ಹಸಿರು ಅಥವಾ ಲಘು ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest mature groups; avoid handling damage.",
            "kn": "ಬೆಳೆದ ಗುಂಪುಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಹಾನಿ ಆಗದಂತೆ ಎಚ್ಚರಿಕೆಯಿಂದ ಹಿಡಿಯಿರಿ."
        },
        "rejuvenation": {
            "en": "Remove old stem after harvest; leave strong follower sucker.",
            "kn": "ಕೊಯ್ತಿನ ನಂತರ ಹಳೆ ದಿಂಡ ತೆಗೆದುಹಾಕಿ; ಬಲವಾದ ಸಕ್ಕರ್‌ನ್ನು ಮಾತ್ರ ಉಳಿಸಿ."
        }
    },
    "grapes": {
        "seedling": {
            "en": "Plant grafted vines; provide strong trellis support; ensure well-drained soil.",
            "kn": "ಗ್ರಾಫ್ಟ್ ಮಾಡಿದ ವೈನ್ ನೆಡಿರಿ; ಬಲವಾದ ಟ್ರೆಲ್ಲಿಸ್ ಒದಗಿಸಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ."
        },
        "vegetative": {
            "en": "Train vines properly; apply nitrogen; maintain irrigation; prune shoots.",
            "kn": "ವೈನ್‌ಗಳನ್ನು ಸರಿಯಾಗಿ ತರಬೇತಿ ಮಾಡಿ; ನೈಟ್ರೋಜನ್ ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ; ಶೂಟ್‌ಗಳನ್ನು ಕತ್ತರಿಸಿ."
        },
        "flowering": {
            "en": "Avoid heavy irrigation; apply boron; protect from downy and powdery mildew.",
            "kn": "ಹೆಚ್ಚು ನೀರಾವರಿ ಬೇಡ; ಬೋರಾನ್ ನೀಡಿ; ಡೌನಿ ಮತ್ತು ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Maintain steady moisture; apply potash; thin bunches for quality.",
            "kn": "ಸಮ ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಗುಣಮಟ್ಟಕ್ಕೆ ಬುಂಚ್ ತೆಳುವಾಗಿಸಿ."
        },
        "maturity": {
            "en": "Sugars accumulate; reduce irrigation; protect from berry cracking.",
            "kn": "ಸಕ್ಕರೆ ಸೇರಿಕೊಳ್ಳುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಬೆರ್ರಿ ಕ್ರ್ಯಾಕಿಂಗ್ ತಡೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest fully ripe bunches; avoid crushing; store in cool shade.",
            "kn": "ಪೂರ್ಣ ಬೆಳೆದ ಬುಂಚ್‌ಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ನಜ್ಜುಗು ತಪ್ಪಿಸಿ; ತಂಪಾದ ನೆರಳಿನಲ್ಲಿ ಇಡಿ."
        },
        "rejuvenation": {
            "en": "Annual pruning essential; apply compost; remove diseased vines.",
            "kn": "ವಾರ್ಷಿಕ ಕತ್ತರಿಸಿಕೆ ಅಗತ್ಯ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ರೋಗಗ್ರಸ್ತ ವೈನ್‌ಗಳನ್ನು ತೆಗೆಯಿರಿ."
        }
    },
    "pomegranate": {
        "seedling": {
            "en": "Plant healthy seedlings; avoid waterlogging; apply FYM at planting.",
            "kn": "ಆರೋಗ್ಯಕರ ಮೊಳಕೆ ನೆಡಿರಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ; ನೆಡುವಾಗ FYM ನೀಡಿ."
        },
        "vegetative": {
            "en": "Encourage branching; prune suckers; apply nitrogen split doses.",
            "kn": "ಶಾಖೆ ಬೆಳವಣಿಗೆ ಉತ್ತೇಜಿಸಿ; ಸಕ್ಕರ್ ಕತ್ತರಿಸಿ; ನೈಟ್ರೋಜನ್ ಹಂತವಾಗಿ ನೀಡಿ."
        },
        "flowering": {
            "en": "Maintain moisture; apply boron; protect flowers from thrips.",
            "kn": "ತೇವ ಇರಲಿ; ಬೋರಾನ್ ನೀಡಿ; ಹೂಗಳಿಗೆ ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain irrigation; protect from fruit borer and oily spot disease.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ; ಫ್ರೂಟ್ ಬೋರರ್ ಮತ್ತು ಓಯ್ಲಿ ಸ್ಪಾಟ್ ರೋಗ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Fruits develop deep color; avoid cracks by uniform irrigation.",
            "kn": "ಹಣ್ಣುಗಳು ಗಾಢ ಬಣ್ಣ ಪಡೆಯುತ್ತವೆ; ಸಮ ನೀರಾವರಿ ಮೂಲಕ ಕ್ರ್ಯಾಕ್ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when fruits turn glossy and hard; cut with secateurs.",
            "kn": "ಹಣ್ಣುಗಳು ಹೊಳೆಯುವ ಮತ್ತು ಗಟ್ಟಿಯಾಗುವಾಗ ಕೊಯ್ಯಿರಿ; ಕತ್ತರಕ ಬಳಸಿರಿ."
        },
        "rejuvenation": {
            "en": "Prune after harvest; remove infected branches; apply manure.",
            "kn": "ಕೊಯ್ತಿನ ನಂತರ ಕತ್ತರಿಸಿ; ಸೋಂಕಿತ ಶಾಖೆ ತೆಗೆಯಿರಿ; ಗೊಬ್ಬರ ನೀಡಿ."
        }
    },
    "papaya": {
        "seedling": {
            "en": "Raise seedlings in polybags; transplant at 45–60 days; avoid waterlogging.",
            "kn": "ಪಾಲಿಬ್ಯಾಗ್‌ಗಳಲ್ಲಿ ನರ್ಸರಿ ಬೆಳೆಸಿರಿ; 45–60 ದಿನದಲ್ಲಿ ನಾಟಿಕೆ ಮಾಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen every month; maintain moisture; control mites and mealybugs.",
            "kn": "ಪ್ರತಿ ತಿಂಗಳು ನೈಟ್ರೋಜನ್ ನೀಡಿ; ತೇವ ಇರಲಿ; ಮೈಟ್ ಮತ್ತು ಮೀಲಿ ಬಗ್ ತಡೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Ensure steady irrigation; apply boron; protect flowers from drooping.",
            "kn": "ನಿಯಮಿತ ನೀರಾವರಿ ಇರಲಿ; ಬೋರಾನ್ ನೀಡಿ; ಹೂ ಬಿದ್ದುಹೋಗುವುದನ್ನು ತಪ್ಪಿಸಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain moisture; ensure nutrient balance.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ತೇವ ಇರಲಿ; ಪೋಷಕಾಂಶ ಸಮತೋಲನ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "maturity": {
            "en": "Fruit skin turns yellow; reduce irrigation.",
            "kn": "ಹಣ್ಣು ತೋಳು ಹಳದಿ ಆಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when 20–30% skin turns yellow; handle gently.",
            "kn": "20–30% ತೋಳು ಹಳದಿ ಆದಾಗ ಕೊಯ್ಯಿರಿ; ಎಚ್ಚರಿಕೆಯಿಂದ ಹಿಡಿಯಿರಿ."
        },
        "rejuvenation": {
            "en": "Remove dried leaves; apply compost; manage spacing.",
            "kn": "ಒಣ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        }
    },
    "guava": {
        "seedling": {
            "en": "Plant grafted seedlings; ensure good drainage; apply FYM in pits.",
            "kn": "ಗ್ರಾಫ್ಟ್ ಮಾಡಿದ ಮೊಳಕೆ ನೆಡಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ; ಗುಂಡಿಯಲ್ಲಿ FYM ನೀಡಿ."
        },
        "vegetative": {
            "en": "Prune water shoots; apply NPK; maintain basin weed-free.",
            "kn": "ವಾಟರ್ ಶೂಟ್‌ಗಳನ್ನು ಕತ್ತರಿಸಿ; NPK ನೀಡಿ; ಬೇಸಿನ್ ಕಳೆ ರಹಿತವಾಗಿಡಿ."
        },
        "flowering": {
            "en": "Ensure moisture; apply boron; control fruit fly and thrips.",
            "kn": "ತೇವ ಇರಲಿ; ಬೋರಾನ್ ನೀಡಿ; ಫ್ರೂಟ್ ಫ್ಲೈ ಮತ್ತು ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain irrigation; thin fruits for quality.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ; ಗುಣಮಟ್ಟಕ್ಕಾಗಿ ಹಣ್ಣು ತೆಳುವಾಗಿ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Fruits soften; reduce irrigation; protect from cracking.",
            "kn": "ಹಣ್ಣು ಮೃದುಗೊಳ್ಳುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಕ್ರ್ಯಾಕ್ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest mature, firm fruits; avoid rough handling.",
            "kn": "ಬೆಳೆದ, ಗಟ್ಟಿಯಾದ ಹಣ್ಣುಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಕಠಿಣವಾಗಿ ಹಿಡಿಯಬೇಡಿ."
        },
        "rejuvenation": {
            "en": "Hard prune once a year; apply manure; remove diseased branches.",
            "kn": "ವರ್ಷಕ್ಕೆ ಒಮ್ಮೆ ಕಠಿಣ ಕತ್ತರಿಸಿ; ಗೊಬ್ಬರ ನೀಡಿ; ರೋಗಗ್ರಸ್ತ ಶಾಖೆ ತೆಗೆಯಿರಿ."
        }
    },
    "sapota": {
        "seedling": {
            "en": "Plant grafted saplings; provide regular irrigation initially.",
            "kn": "ಗ್ರಾಫ್ಟ್ ಮಾಡಿದ ಮೊಳಕೆ ನೆಡಿರಿ; ಆರಂಭದಲ್ಲಿ ನಿಯಮಿತ ನೀರಾವರಿ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply NPK; prune water shoots; maintain weed-free basins.",
            "kn": "NPK ನೀಡಿ; ವಾಟರ್ ಶೂಟ್ ಕತ್ತರಿಸಿ; ಬೇಸಿನ್ ಸ್ವಚ್ಛ ಇರಲಿ."
        },
        "flowering": {
            "en": "Ensure steady moisture; apply boron; protect from sapota seed borer.",
            "kn": "ನಿಯಮಿತ ತೇವ ಇರಲಿ; ಬೋರಾನ್ ನೀಡಿ; ಸಪೋಟ ಬೀಜ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain moisture; avoid heavy irrigation.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ನೀರಾವರಿ ಬೇಡ."
        },
        "maturity": {
            "en": "Fruits turn light brown; reduce irrigation.",
            "kn": "ಹಣ್ಣು ಲಘು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when fruits are firm but mature; check latex flow.",
            "kn": "ಹಣ್ಣು ಗಟ್ಟಿಯಾಗಿದ್ದು ಬೆಳೆದಾಗ ಕೊಯ್ಯಿರಿ; ಲ್ಯಾಟೆಕ್ಸ್ ಹರಿವು ಪರಿಶೀಲಿಸಿ."
        },
        "rejuvenation": {
            "en": "Annual pruning; apply manure; remove pest-affected branches.",
            "kn": "ವಾರ್ಷಿಕ ಕತ್ತರಿಸಿಕೆ; ಗೊಬ್ಬರ ನೀಡಿ; ಕೀಟಬಾಧಿತ ಶಾಖೆ ತೆಗೆಯಿರಿ."
        }
    },
    "orange": {
        "seedling": {
            "en": "Use disease-free grafts; plant in well-drained soil; provide shade initially.",
            "kn": "ರೋಗರಹಿತ ಗ್ರಾಫ್ಟ್ ಬಳಸಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣಿನಲ್ಲಿ ನೆಡಿರಿ; ಆರಂಭದಲ್ಲಿ ನೆರಳು ಒದಗಿಸಿ."
        },
        "vegetative": {
            "en": "Apply NPK; prune water sprouts; maintain basin moisture.",
            "kn": "NPK ನೀಡಿ; ವಾಟರ್ ಸ್ಪ್ರೌಟ್‌ಗಳನ್ನು ಕತ್ತರಿಸಿ; ಬೇಸಿನ್ ತೇವ ಇರಲಿ."
        },
        "flowering": {
            "en": "Light irrigation; apply micronutrients; protect blooms from pests.",
            "kn": "ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಹೂಗಳಿಗೆ ಕೀಟ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Maintain moisture; apply potash; thin fruits for size improvement.",
            "kn": "ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಹಣ್ಣಿನ ಗಾತ್ರಕ್ಕೆ ತೆಳು ಮಾಡಿರಿ."
        },
        "maturity": {
            "en": "Fruits change color; reduce irrigation; avoid cracking.",
            "kn": "ಹಣ್ಣು ಬಣ್ಣ ಬದಲಾಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಕ್ರ್ಯಾಕ್ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when fully colored; avoid pulling fruits by hand.",
            "kn": "ಪೂರ್ಣ ಬಣ್ಣ ಆದಾಗ ಕೊಯ್ಯಿರಿ; ಹಸ್ತದಿಂದ ಎಳೆಯಬೇಡಿ."
        },
        "rejuvenation": {
            "en": "Prune old branches; apply compost; maintain tree shape.",
            "kn": "ಹಳೆಯ ಶಾಖೆಗಳನ್ನು ಕತ್ತರಿಸಿ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ಮರದ ಆಕರವನ್ನು ಕಾಯ್ದುಕೊಳ್ಳಿ."
        }
    },
    "onion": {
        "nursery": {
            "en": "Raise seedlings in well-prepared beds; apply FYM; keep moist but not waterlogged.",
            "kn": "ಸರಿ ತಯಾರು ಬೆಡ್‌ನಲ್ಲಿ ನರ್ಸರಿ ಬೆಳೆಸಿ; FYM ನೀಡಿ; ತೇವ ಇರಲಿ ಆದರೆ ಜಲಾವೃತ ಬೇಡ."
        },
        "transplanting": {
            "en": "Transplant 40–45 day seedlings; plant shallow; irrigate lightly.",
            "kn": "40–45 ದಿನದ ಮೊಳಕೆ ನಾಟಿಕೆ ಮಾಡಿ; ಅಲ್ಪ ಆಳದಲ್ಲಿ ನೆಡಿರಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; ensure weed-free field; maintain regular irrigation.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ರಹಿತ ಹೊಲ ಇರಲಿ; ನಿಯಮಿತ ನೀರಾವರಿ ಮಾಡಿ."
        },
        "bulb_initiation": {
            "en": "Apply potash; ensure uniform moisture; avoid waterlogging.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ಸಮ ತೇವ ಇರಲಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "bulb_development": {
            "en": "Reduce nitrogen; maintain light irrigation; protect from thrips.",
            "kn": "ನೈಟ್ರೋಜನ್ ಕಡಿಮೆ ಮಾಡಿ; ಲಘು ನೀರಾವರಿ ಇರಲಿ; ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when tops dry and fall; cure bulbs before storage.",
            "kn": "ಎಲೆಗಳು ಒಣಗಿ ಬಿದ್ದಾಗ ಕೊಯ್ಯಿರಿ; ಸಂಗ್ರಹಣೆಗೆ ಮುಂಚೆ ಬಲ್ಪ್‌ಗಳನ್ನು ಕ್ಯೂರ್ ಮಾಡಿ."
        }
    },

    "tomato": {
        "nursery": {
            "en": "Raise seedlings in trays; treat seeds; protect from damping-off.",
            "kn": "ಟ್ರೇಗಳಲ್ಲಿ ನರ್ಸರಿ ಬೆಳೆಸಿ; ಬೀಜ ಶೋಧಿಸಿ; ಡ್ಯಾಂಪಿಂಗ್-ಆಫ್ ತಡೆ ಮಾಡಿ."
        },
        "transplanting": {
            "en": "Transplant 25–30 day seedlings; provide stakes; irrigate lightly.",
            "kn": "25–30 ದಿನದ ಮೊಳಕೆ ನಾಟಿಕೆ ಮಾಡಿ; ಸ್ಟೇಕ್ ನೀಡಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; prune lower leaves; maintain weed-free field.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕೆಳಗಿನ ಎಲೆ ಕತ್ತರಿಸಿ; ಹೊಲ ಕಳೆ ರಹಿತವಾಗಿಡಿ."
        },
        "flowering": {
            "en": "Apply boron; maintain moisture; avoid heavy nitrogen.",
            "kn": "ಬೋರಾನ್ ನೀಡಿ; ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ."
        },
        "fruiting": {
            "en": "Apply potash; prevent fruit borer; maintain irrigation interval.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ಫ್ರೂಟ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ; ನೀರಾವರಿ ಕ್ರಮ ಪಾಲಿಸಿ."
        },
        "harvest": {
            "en": "Harvest at breaker or pink stage depending on market; handle gently.",
            "kn": "ಮಾರ್ಕೆಟ್‌ಗೆ ಅನುಸಾರ ಬ್ರೇಕರ್/ಪಿಂಕ್ ಹಂತದಲ್ಲಿ ಕೊಯ್ಯಿರಿ; ಎಚ್ಚರಿಕೆಯಿಂದ ಹಿಡಿಯಿರಿ."
        }
    },
    "potato": {
        "planting": {
            "en": "Use disease-free seed tubers; plant in cool season; ensure good drainage.",
            "kn": "ರೋಗರಹಿತ ಕಾಯಿ ಬೀಜ ಬಳಸಿ; ಶೀತ ಋತುವಿನಲ್ಲಿ ನೆಡಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ."
        },
        "sprout_emergence": {
            "en": "Light irrigation; protect sprouts from frost; apply basal fertilizers.",
            "kn": "ಲಘು ನೀರಾವರಿ; ಫ್ರಾಸ್ಟ್ ತಡೆ; ಮೂಲ ಗೊಬ್ಬರ ನೀಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; earth-up plants; maintain moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಣ್ಣು ಬೋಡು ಮಾಡಿ; ತೇವ ಇರಲಿ."
        },
        "tuber_initiation": {
            "en": "Reduce nitrogen; ensure regular irrigation; apply potash.",
            "kn": "ನೈಟ್ರೋಜನ್ ಕಡಿಮೆ ಮಾಡಿ; ನಿಯಮಿತ ನೀರಾವರಿ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "tuber_bulking": {
            "en": "Critical stage; maintain constant moisture; avoid waterlogging.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ಸಮ ತೇವ ಇರಲಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when vines dry; cure tubers before storage.",
            "kn": "ವೈನ್ ಒಣಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಸಂಗ್ರಹಣೆಗೆ ಮುಂಚೆ ಕಾಯಿ ಕ್ಯೂರ್ ಮಾಡಿ."
        }
    },

    "brinjal": {
        "nursery": {
            "en": "Raise seedlings in well-prepared beds; treat seeds; protect from damping-off.",
            "kn": "ನರ್ಸರಿ ಬೆಡ್‌ನಲ್ಲಿ ಮೊಳಕೆ ಬೆಳೆಸಿ; ಬೀಜ ಶೋಧಿಸಿ; ಡ್ಯಾಂಪಿಂಗ್-ಆಫ್ ತಡೆ ಮಾಡಿ."
        },
        "transplanting": {
            "en": "Transplant 30–35 day seedlings; irrigate lightly; provide spacing.",
            "kn": "30–35 ದಿನದ ಮೊಳಕೆ ನಾಟಿಕೆ ಮಾಡಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; prune side shoots; maintain weed-free field.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಬದಿ ಶೂಟ್‌ಗಳನ್ನು ಕತ್ತರಿಸಿ; ಹೊಲ ಕಳೆ ರಹಿತವಾಗಿಡಿ."
        },
        "flowering": {
            "en": "Maintain moisture; apply boron; protect from shoot & fruit borer.",
            "kn": "ತೇವ ಇರಲಿ; ಬೋರಾನ್ ನೀಡಿ; ಶೂಟ್ ಮತ್ತು ಫ್ರೂಟ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain irrigation; harvest regularly.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ; ನಿಯಮಿತವಾಗಿ ಕೊಯ್ಯಿರಿ."
        },
        "harvest": {
            "en": "Harvest tender fruits; avoid over-mature hard fruits.",
            "kn": "ಮೃದುವಾದ ಹಣ್ಣು ಕೊಯ್ಯಿರಿ; ಹೆಚ್ಚು ಬೆಳೆದ ಗಟ್ಟಿಯಾದ ಹಣ್ಣು ತಪ್ಪಿಸಿ."
        }
    },

    "chilli": {
        "nursery": {
            "en": "Raise seedlings in trays; treat seeds; protect from fungal damping-off.",
            "kn": "ಟ್ರೇಗಳಲ್ಲಿ ನರ್ಸರಿ ಮಾಡಿ; ಬೀಜ ಶೋಧಿಸಿ; ಫಂಗಲ್ ಡ್ಯಾಂಪಿಂಗ್-ಆಫ್ ತಡೆ ಮಾಡಿ."
        },
        "transplanting": {
            "en": "Transplant 35–40 day seedlings; irrigate lightly; allow good aeration.",
            "kn": "35–40 ದಿನದ ಮೊಳಕೆ ನಾಟಿಕೆ ಮಾಡಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಗಾಳಿ ಹರಿವು ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; remove suckers; avoid waterlogging.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಸಕ್ಕರ್ ತೆಗೆದುಹಾಕಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Avoid excess nitrogen; apply potash; protect flowers from thrips.",
            "kn": "ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ಬೇಡ; ಪೊಟಾಶ್ ನೀಡಿ; ಹೂಗಳಿಗೆ ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Maintain irrigation; apply micronutrients; monitor for mites.",
            "kn": "ನೀರಾವರಿ ಇರಲಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಮೈಟ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest red or green fruits depending on purpose; avoid wet harvesting.",
            "kn": "ಉದ್ದೇಶಕ್ಕೆ ಅನುಸಾರ ಹಸಿರು/ಕೆಂಪು ಹಣ್ಣು ಕೊಯ್ಯಿರಿ; ಒದ್ದೆಯಾದಾಗ ಕೊಯ್ಯಬೇಡಿ."
        }
    },

    "cabbage": {
        "nursery": {
            "en": "Raise seedlings in cool climate; treat seeds; maintain moisture.",
            "kn": "ಶೀತ ಹವಾಮಾನದಲ್ಲಿ ನರ್ಸರಿ ಬೆಳೆಸಿ; ಬೀಜ ಶೋಧಿಸಿ; ತೇವ ಇರಲಿ."
        },
        "transplanting": {
            "en": "Transplant 25–30 day seedlings; provide spacing and light irrigation.",
            "kn": "25–30 ದಿನದ ಮೊಳಕೆ ನಾಟಿಕೆ ಮಾಡಿ; ಅಂತರ ಮತ್ತು ಲಘು ನೀರಾವರಿ ನೀಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; ensure weed-free field; maintain regular irrigation.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ರಹಿತ ಹೊಲ ಇರಲಿ; ನಿಯಮಿತ ನೀರಾವರಿ ಮಾಡಿ."
        },
        "heading": {
            "en": "Critical stage; maintain moisture; apply potash; avoid waterlogging.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Heads become compact; reduce irrigation.",
            "kn": "ಹೆಡ್ ಗಟ್ಟಿ ಆಗುತ್ತದೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when heads are firm; avoid splitting.",
            "kn": "ಹೆಡ್ ಗಟ್ಟಿ ಆದಾಗ ಕೊಯ್ಯಿರಿ; ಸ್ಪ್ಲಿಟ್ ಆಗದಂತೆ ನೋಡಿಕೊಳ್ಳಿ."
        }
    },

    "cauliflower": {
        "nursery": {
            "en": "Raise seedlings in cool months; treat seeds; protect from damping-off.",
            "kn": "ಶೀತ ಋತುವಿನಲ್ಲಿ ನರ್ಸರಿ ಮಾಡಿ; ಬೀಜ ಶೋಧಿಸಿ; ಡ್ಯಾಂಪಿಂಗ್-ಆಫ್ ತಡೆ ಮಾಡಿ."
        },
        "transplanting": {
            "en": "Transplant 30-day seedlings; irrigate lightly; provide spacing.",
            "kn": "30 ದಿನದ ಮೊಳಕೆ ನಾಟಿಕೆ ಮಾಡಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ; ಅಂತರ ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; maintain irrigation; keep field weed-free.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ; ಹೊಲ ಕಳೆ ರಹಿತವಾಗಿಡಿ."
        },
        "curd_initiation": {
            "en": "Maintain uniform moisture; avoid heat stress; apply boron.",
            "kn": "ಸಮತೋಲನ ತೇವ ಇರಲಿ; ಬಿಸಿ ಒತ್ತಡ ತಪ್ಪಿಸಿ; ಬೋರಾನ್ ನೀಡಿ."
        },
        "curd_development": {
            "en": "Provide shade to curds if sun is strong; maintain moisture.",
            "kn": "ಬಿಸಿಲು ಜಾಸ್ತಿ ಇದ್ದರೆ ಕರ್ಡ್‌ಗೆ ನೆರಳು ಒದಗಿಸಿ; ತೇವ ಇರಲಿ."
        },
        "harvest": {
            "en": "Harvest when curds are compact and white; avoid yellowing.",
            "kn": "ಕರ್ಡ್ ಗಟ್ಟಿ ಮತ್ತು ಬಿಳಿಯಾಗಿದ್ದಾಗ ಕೊಯ್ಯಿರಿ; ಹಳದಿ ಆಗದಂತೆ ನೋಡಿಕೊಳ್ಳಿ."
        }
    },

    "beans": {
        "planting": {
            "en": "Direct sow seeds; ensure well-drained soil; apply FYM.",
            "kn": "ನೆರಳಾಗಿ ಬೀಜ ಬಿತ್ತಿರಿ; ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣು ಇರಲಿ; FYM ನೀಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen lightly; provide staking for climbers; maintain moisture.",
            "kn": "ಲಘು ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಏರುವ ತಳಿಗಳಿಗೆ ಸ್ಟೇಕ್ ನೀಡಿ; ತೇವ ಇರಲಿ."
        },
        "flowering": {
            "en": "Apply micronutrients; avoid moisture stress; protect from thrips.",
            "kn": "ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ತೇವ ಕೊರತೆ ತಪ್ಪಿಸಿ; ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "pod_development": {
            "en": "Maintain irrigation; apply potash; harvest regularly.",
            "kn": "ನೀರಾವರಿ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ; ನಿಯಮಿತವಾಗಿ ಕೊಯ್ಯಿರಿ."
        },
        "maturity": {
            "en": "Reduce irrigation; pods mature and fill fully.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಪೊಡ್‌ಗಳು ಪೂರ್ಣಗೊಳ್ಳುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest tender pods; avoid fibrous old pods.",
            "kn": "ಮೃದುವಾದ ಪೊಡ್‌ಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಹಳೆಯ ಗಟ್ಟಿ ಪೊಡ್ ತಪ್ಪಿಸಿ."
        }
    },
    "cucumber": {
        "planting": {
            "en": "Direct sow seeds; ensure light irrigation; apply FYM.",
            "kn": "ನೆರಳಾಗಿ ಬೀಜ ಬಿತ್ತಿರಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ; FYM ನೀಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; provide trellis support; maintain moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಟ್ರೆಲ್ಲಿಸ್ ಒದಗಿಸಿ; ತೇವ ಇರಲಿ."
        },
        "flowering": {
            "en": "Maintain moisture; apply boron for fruit set; protect from thrips.",
            "kn": "ತೇವ ಇರಲಿ; ಫ್ರೂಟ್ ಸೆಟ್‌ಗೆ ಬೋರಾನ್ ನೀಡಿ; ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain regular irrigation; harvest tender fruits.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನಿಯಮಿತ ನೀರಾವರಿ; ಮೃದುವಾದ ಹಣ್ಣು ಕೊಯ್ಯಿರಿ."
        },
        "maturity": {
            "en": "Reduce irrigation; fruits turn light yellow or size stabilizes.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಹಣ್ಣು ಲಘು ಹಳದಿ/ಗಾತ್ರ ಸ್ಥಿರವಾಗುತ್ತದೆ."
        },
        "harvest": {
            "en": "Harvest early morning for freshness; avoid overmature fruits.",
            "kn": "ತಾಜಾಗಿರಲು ಬೆಳಗ್ಗೆ ಕೊಯ್ಯಿರಿ; ಹೆಚ್ಚು ಬೆಳೆದ ಹಣ್ಣು ತಪ್ಪಿಸಿ."
        }
    },
    "turmeric": {
        "planting": {
            "en": "Use healthy rhizomes; treat with fungicide; plant in raised beds with good drainage.",
            "kn": "ಆರೋಗ್ಯಕರ ಬೇರುಗಳನ್ನು ಬಳಸಿ; ಫಂಗಿಸೈಡ್ ಶೋಧಿಸಿ; ಚೆನ್ನಾಗಿ ನೀರು ಹೋದ ಎತ್ತರದ ಬೆಡ್‌ಗಳಲ್ಲಿ ನೆಡಿರಿ."
        },
        "sprouting": {
            "en": "Maintain light irrigation; apply FYM; keep soil loose for shoot emergence.",
            "kn": "ಲಘು ನೀರಾವರಿ ಇರಲಿ; FYM ನೀಡಿ; ಮೊಳಕೆ ಹೊರಬರಲು ಮಣ್ಣು ಸಡಿಲ ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen and potash; mulch to retain moisture; control leaf spot.",
            "kn": "ನೈಟ್ರೋಜನ್ ಮತ್ತು ಪೊಟಾಶ್ ನೀಡಿ; ತೇವ ಉಳಿಸಲು ಮಲ್ಚ್ ಮಾಡಿ; ಲೀಫ್ ಸ್ಪಾಟ್ ತಡೆ ಮಾಡಿ."
        },
        "rhizome_development": {
            "en": "Ensure steady moisture; apply second split fertilizers; avoid waterlogging.",
            "kn": "ನಿಯಮಿತ ತೇವ ಇರಲಿ; ಎರಡನೇ ಹಂತದ ಗೊಬ್ಬರ ನೀಡಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Leaves begin yellowing; reduce irrigation gradually.",
            "kn": "ಎಲೆಗಳು ಹಳದಿ ಆಗಲು ಪ್ರಾರಂಭ; ನಿಧಾನವಾಗಿ ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when leaves dry; clean and cure rhizomes properly.",
            "kn": "ಎಲೆಗಳು ಒಣಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಬೇರುಗಳನ್ನು ಸ್ವಚ್ಛವಾಗಿ ಕ್ಯೂರ್ ಮಾಡಿ."
        }
    },
    "ginger": {
        "planting": {
            "en": "Use disease-free rhizomes; plant in moist, well-drained soil; apply FYM.",
            "kn": "ರೋಗರಹಿತ ಬೇರುಗಳನ್ನು ಬಳಸಿ; ತೇವಯುತ, ನೀರು ಚೆನ್ನಾಗಿ ಹೋದ ಮಣ್ಣಿನಲ್ಲಿ ನೆಡಿರಿ; FYM ನೀಡಿ."
        },
        "sprouting": {
            "en": "Light irrigation; mulch to retain moisture; protect from shoot borer.",
            "kn": "ಲಘು ನೀರಾವರಿ; ತೇವ ಉಳಿಸಲು ಮಲ್ಚ್ ಮಾಡಿ; ಶೂಟ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; keep field weed-free; maintain moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಕಳೆ ರಹಿತ ಹೊಲ ಇರಲಿ; ತೇವ ಇರಲಿ."
        },
        "rhizome_development": {
            "en": "Apply potash; maintain steady moisture; avoid standing water.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ತೇವ ನಿರಂತರ ಇರಲಿ; ನಿಂತ ನೀರು ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Lower leaves turn yellow; reduce irrigation.",
            "kn": "ಕೆಳಗಿನ ಎಲೆಗಳು ಹಳದಿ ಆಗುತ್ತವೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when rhizomes become plump and aromatic.",
            "kn": "ಬೇರುಗಳು ಗಟ್ಟಿಯಾಗಿದ್ದು ವಾಸನೆ ಬರಲು ಪ್ರಾರಂಭವಾದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },
    "coriander": {
        "sowing": {
            "en": "Sow split seeds; ensure fine tilth; irrigate lightly.",
            "kn": "ಪಿಂಗಾಣಿಸಿದ ಬೀಜ ಬಿತ್ತಿರಿ; ಉತ್ತಮ ಮಣ್ಣು ತಯಾರಿ ಇರಲಿ; ಲಘು ನೀರಾವರಿ ಮಾಡಿ."
        },
        "vegetative": {
            "en": "Apply light nitrogen; maintain regular irrigation; avoid waterlogging.",
            "kn": "ಲಘು ನೈಟ್ರೋಜನ್ ನೀಡಿ; ನಿಯಮಿತ ನೀರಾವರಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Stop nitrogen; apply micronutrients; maintain even moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನಿಲ್ಲಿಸಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ಸಮತೋಲನ ತೇವ ಇರಲಿ."
        },
        "seed_development": {
            "en": "Reduce irrigation; protect from aphids; avoid humidity stress.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಆಫಿಡ್ ತಡೆ ಮಾಡಿ; ತೇವ ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when seeds turn brown; dry well before storage.",
            "kn": "ಬೀಜಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿದಾಗ ಕೊಯ್ಯಿರಿ; ಸಂಗ್ರಹಣೆಗೆ ಮುಂಚೆ ಚೆನ್ನಾಗಿ ಒಣಗಿಸಿ."
        }
    },

    "pepper": {
        "planting": {
            "en": "Plant healthy vines near live standards; ensure shade and moisture.",
            "kn": "ಆರೋಗ್ಯಕರ ವೈನ್‌ಗಳನ್ನು ಜೀವಂತ ಮರಗಳ ಬಳಿ ನೆಡಿರಿ; ನೆರಳು ಮತ್ತು ತೇವ ಇರಲಿ."
        },
        "vegetative": {
            "en": "Train vines; apply compost; protect from insects and wilt disease.",
            "kn": "ವೈನ್‌ಗಳಿಗೆ ತರಬೇತಿ ನೀಡಿ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ; ಕೀಟ ಮತ್ತು ವಿಲ್ಟ್ ರೋಗ ತಡೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Ensure steady moisture; apply micronutrients; avoid moisture stress.",
            "kn": "ತೇವ ನಿರಂತರ ಇರಲಿ; ಸೂಕ್ಷ್ಮಾಂಶ ನೀಡಿ; ತೇವ ಕೊರತೆ ತಪ್ಪಿಸಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain shade; protect from pollu beetle.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೆರಳು ಇರಲಿ; ಪೋಲ್ಲು ಬಿಟ್ಟಲ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Berries turn red or dark; reduce irrigation.",
            "kn": "ಬೆರ್ರಿಗಳು ಕೆಂಪು/ಕತ್ತಲೆ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ; ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest mature spikes; dry well for black pepper quality.",
            "kn": "ಬೆಳೆದ ಸ್ಪೈಕ್‌ಗಳನ್ನು ಕೊಯ್ಯಿರಿ; ಉತ್ತಮ ಬ್ಲಾಕ್ ಪೆಪ್ಪರ್‌ಗೆ ಚೆನ್ನಾಗಿ ಒಣಗಿಸಿ."
        }
    },

    "cardamom": {
        "planting": {
            "en": "Plant suckers in shaded, cool, moist areas; apply compost.",
            "kn": "ನೆರಳು, ತಂಪು ಮತ್ತು ತೇವ ಪ್ರದೇಶದಲ್ಲಿ ಸಕ್ಕರ್ ನೆಡಿರಿ; ಕಂಪೋಸ್ಟ್ ನೀಡಿ."
        },
        "vegetative": {
            "en": "Maintain shade; apply nitrogen; keep soil moist; remove dried shoots.",
            "kn": "ನೆರಳು ಇರಲಿ; ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಣ್ಣು ತೇವ ಇರಲಿ; ಒಣ ಶೂಟ್‌ಗಳನ್ನು ತೆಗೆಯಿರಿ."
        },
        "flowering": {
            "en": "Ensure constant moisture; protect panicles from thrips.",
            "kn": "ನಿಯಮಿತ ತೇವ ಇರಲಿ; ಪಾನಿಕಲ್‌ಗಳನ್ನು ಥ್ರಿಪ್ಸ್‌ನಿಂದ ರಕ್ಷಿಸಿ."
        },
        "capsule_development": {
            "en": "Apply potash; maintain shade and irrigation; monitor for capsule borer.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೆರಳು ಮತ್ತು ನೀರಾವರಿ ಇರಲಿ; ಕ್ಯಾಪ್ಸುಲ್ ಬೋರರ್ ತಡೆ ಮಾಡಿ."
        },
        "maturity": {
            "en": "Capsules turn greenish-yellow; reduce irrigation gradually.",
            "kn": "ಕ್ಯಾಪ್ಸುಲ್ ಹಸಿರು-ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತದೆ; ನಿಧಾನವಾಗಿ ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Handpick mature capsules; dry in controlled conditions for best aroma.",
            "kn": "ಬೆಳೆದ ಕ್ಯಾಪ್ಸುಲ್‌ಗಳನ್ನು ಕೈಯಿಂದ ಕೊಯ್ಯಿರಿ; ಉತ್ತಮ ವಾಸನೆಗೆ ನಿಯಂತ್ರಿತ ಒಣಗಿಸುವಿಕೆ ಮಾಡಿ."
        }
    },










 





















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

    # -------------------------
    # FOOD GRAINS – CEREALS
    # -------------------------
    "paddy": {
        "nursery": (30, 15, 15),
        "vegetative": (50, 20, 20),
        "tillering": (40, 0, 20),
        "panicle": (30, 0, 30),
        "maturity": (0, 0, 0)
    },

    "ragi": {
        "vegetative": (40, 20, 20),
        "tillering": (20, 0, 20),
        "flowering": (10, 0, 20)
    },

    "jowar": {
        "vegetative": (40, 20, 0),
        "flowering": (20, 0, 20)
    },

    "maize": {
        "vegetative": (80, 40, 20),
        "knee_high": (40, 0, 20),
        "tasseling": (20, 0, 20)
    },

    "bajra": {
        "vegetative": (40, 20, 0),
        "flowering": (20, 0, 20)
    },

    "wheat": {
        "vegetative": (60, 40, 20),
        "tillering": (30, 0, 20),
        "flowering": (20, 0, 20)
    },

    # -------------------------
    # PULSES
    # -------------------------
    "pigeon_pea": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20)
    },

    "green_gram": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20)
    },

    "black_gram": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20)
    },

    "bengal_gram": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 0, 20)
    },

    "horse_gram": {
        "vegetative": (15, 20, 10)
    },

    "cowpea": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 0, 20)
    },

    # -------------------------
    # OILSEEDS
    # -------------------------
    "groundnut": {
        "vegetative": (20, 40, 40),
        "flowering": (10, 0, 40)
    },

    "sunflower": {
        "vegetative": (40, 40, 20),
        "flowering": (20, 20, 20)
    },

    "soybean": {
        "vegetative": (20, 60, 40)
    },

    "sesame": {
        "vegetative": (30, 20, 20),
        "flowering": (10, 0, 20)
    },

    "castor": {
        "vegetative": (40, 40, 0),
        "flowering": (20, 20, 20)
    },

    "safflower": {
        "vegetative": (40, 20, 20)
    },

    # -------------------------
    # COMMERCIAL CROPS
    # -------------------------
    "sugarcane": {
        "planting": (60, 40, 40),
        "vegetative": (80, 20, 40),
        "grand_growth": (80, 20, 80)
    },

    "cotton": {
        "vegetative": (40, 20, 20),
        "square_formation": (20, 10, 20),
        "boll_formation": (20, 0, 40)
    },

    "tobacco": {
        "vegetative": (40, 40, 20),
        "leaf_expansion": (20, 20, 40)
    },

    # -------------------------
    # PLANTATION CROPS
    # -------------------------
    "coffee": {
        "vegetative": (40, 40, 40),
        "flowering": (20, 20, 20),
        "fruiting": (40, 20, 40)
    },

    "tea": {
        "vegetative": (60, 40, 40),
        "shoot_growth": (40, 20, 40)
    },

    "arecanut": {
        "vegetative": (100, 40, 140),
        "flowering": (40, 20, 60)
    },

    "coconut": {
        "vegetative": (100, 40, 140),
        "flowering": (40, 20, 60)
    },

    "rubber": {
        "vegetative": (30, 30, 30)
    },

    "cashew": {
        "vegetative": (40, 20, 40),
        "flowering": (20, 10, 20)
    },

    # -------------------------
    # FRUITS
    # -------------------------
    "mango": {
        "vegetative": (60, 40, 40),
        "flowering": (20, 20, 40),
        "fruiting": (20, 0, 40)
    },

    "banana": {
        "vegetative": (100, 40, 60),
        "flowering": (50, 20, 100),
        "fruiting": (40, 20, 60)
    },

    "grapes": {
        "vegetative": (60, 40, 40),
        "flowering": (20, 20, 20),
        "fruiting": (40, 20, 40)
    },

    "pomegranate": {
        "vegetative": (40, 20, 20),
        "flowering": (20, 20, 40),
        "fruiting": (20, 0, 40)
    },

    "papaya": {
        "vegetative": (60, 40, 40),
        "flowering": (20, 20, 20)
    },

    "guava": {
        "vegetative": (40, 20, 20),
        "flowering": (20, 20, 40)
    },

    "sapota": {
        "vegetative": (40, 20, 40),
        "fruiting": (20, 20, 40)
    },

    "orange": {
        "vegetative": (60, 40, 40),
        "flowering": (20, 20, 40),
        "fruiting": (20, 0, 40)
    },

    # -------------------------
    # VEGETABLES
    # -------------------------
    "onion": {
        "vegetative": (40, 20, 20),
        "bulb_initiation": (20, 0, 20),
        "bulb_development": (20, 0, 20)
    },

    "tomato": {
        "vegetative": (60, 40, 20),
        "flowering": (20, 20, 20),
        "fruiting": (20, 0, 40)
    },

    "potato": {
        "vegetative": (60, 40, 40),
        "tuber_initiation": (20, 0, 40),
        "tuber_bulking": (20, 0, 40)
    },

    "brinjal": {
        "vegetative": (40, 20, 20),
        "flowering": (20, 20, 20),
        "fruiting": (20, 0, 40)
    },

    "cabbage": {
        "vegetative": (60, 40, 20),
        "heading": (20, 0, 40)
    },

    "cauliflower": {
        "vegetative": (60, 40, 20),
        "curd_initiation": (20, 20, 20)
    },

    "beans": {
        "vegetative": (20, 40, 20),
        "pod_development": (10, 0, 20)
    },

    "cucumber": {
        "vegetative": (40, 20, 20),
        "flowering": (20, 20, 20),
        "fruiting": (20, 0, 20)
    },

    # -------------------------
    # SPICES
    # -------------------------
    "turmeric": {
        "vegetative": (40, 40, 40),
        "rhizome_development": (20, 20, 40)
    },

    "ginger": {
        "vegetative": (40, 40, 20),
        "rhizome_development": (20, 20, 20)
    },

    "coriander": {
        "vegetative": (20, 40, 20),
        "seed_development": (10, 0, 20)
    },

    "pepper": {
        "vegetative": (20, 20, 40),
        "fruiting": (20, 0, 40)
    },

    "cardamom": {
        "vegetative": (20, 40, 20),
        "capsule_development": (20, 20, 40)
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

    # -------------------------
    # SUCKING PESTS
    # -------------------------
    "aphid": {
        "en": "Spray neem oil 5ml/L or Imidacloprid 0.3ml/L.",
        "kn": "5ml/L ನೀಮ್ ಎಣ್ಣೆ ಅಥವಾ Imidacloprid 0.3ml/L ಸಿಂಪಡಿಸಿ."
    },

    "thrips": {
        "en": "Use Fipronil 2ml/L or neem oil. Maintain field hygiene.",
        "kn": "Fipronil 2ml/L ಅಥವಾ ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ. ಹೊಲ ಸ್ವಚ್ಛತೆ ಇರಲಿ."
    },

    "whitefly": {
        "en": "Spray Imidacloprid 0.3ml/L or Thiamethoxam 0.2g/L.",
        "kn": "Imidacloprid 0.3ml/L ಅಥವಾ Thiamethoxam 0.2g/L ಸಿಂಪಡಿಸಿ."
    },

    "jassid": {
        "en": "Use Acetamiprid 0.2g/L or neem oil.",
        "kn": "Acetamiprid 0.2g/L ಅಥವಾ ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ."
    },

    "mites": {
        "en": "Use Abamectin 0.5ml/L or wettable sulphur.",
        "kn": "Abamectin 0.5ml/L ಅಥವಾ ವೆಟೆಬಲ್ ಸಲ್ಫರ್ ಬಳಸಿ."
    },

    # -------------------------
    # BORERS
    # -------------------------
    "stem borer": {
        "en": "Use Chlorantraniliprole 0.3ml/L or Carbofuran granules in soil.",
        "kn": "Chlorantraniliprole 0.3ml/L ಅಥವಾ Carbofuran ಗ್ರಾನ್ಯುಲ್ಸ್ ಮಣ್ಣಿನಲ್ಲಿ ಬಳಸಿ."
    },

    "fruit borer": {
        "en": "Use Spinosad 0.3ml/L or Emamectin Benzoate 0.4g/L.",
        "kn": "Spinosad 0.3ml/L ಅಥವಾ Emamectin Benzoate 0.4g/L ಸಿಂಪಡಿಸಿ."
    },

    "shoot and fruit borer": {
        "en": "Remove affected shoots; spray Emamectin or Spinosad.",
        "kn": "ಬಾಧಿತ ಶೂಟ್ ತೆಗೆದುಹಾಕಿ; Emamectin ಅಥವಾ Spinosad ಸಿಂಪಡಿಸಿ."
    },

    "pink bollworm": {
        "en": "Spray Emamectin Benzoate or Chlorantraniliprole; use pheromone traps.",
        "kn": "Emamectin Benzoate ಅಥವಾ Chlorantraniliprole ಸಿಂಪಡಿಸಿ; ಫೆರೊಮೊನ್ ಟ್ರ್ಯಾಪ್ ಬಳಸಿ."
    },

    "rice stem borer": {
        "en": "Apply Cartap Hydrochloride 4g/L or Chlorantraniliprole.",
        "kn": "Cartap Hydrochloride 4g/L ಅಥವಾ Chlorantraniliprole ಬಳಸಿ."
    },

    "coffee berry borer": {
        "en": "Use Beauveria bassiana spray; maintain field sanitation.",
        "kn": "Beauveria bassiana ಸಿಂಪಡಿಸಿ; ಹೊಲ ಸ್ವಚ್ಛತೆ ಪಾಲಿಸಿ."
    },

    "banana pseudostem borer": {
        "en": "Apply Carbofuran granules near pseudostem; use pheromone traps.",
        "kn": "Pseudostem ಹತ್ತಿರ Carbofuran ಬಳಸಿ; ಫೆರೊಮೊನ್ ಟ್ರ್ಯಾಪ್ ಬಳಸಿ."
    },

    # -------------------------
    # CATERPILLARS / LEAF-EATING LARVAE
    # -------------------------
    "armyworm": {
        "en": "Spray Chlorantraniliprole 0.3ml/L or Emamectin 0.4g/L.",
        "kn": "Chlorantraniliprole 0.3ml/L ಅಥವಾ Emamectin 0.4g/L ಸಿಂಪಡಿಸಿ."
    },

    "cutworm": {
        "en": "Use Chlorpyrifos 2.5ml/L in soil; remove weeds.",
        "kn": "ಮಣ್ಣಿನಲ್ಲಿ Chlorpyrifos 2.5ml/L ಬಳಸಿ; ಕಳೆ ತೆಗೆಯಿರಿ."
    },

    "leaf folder": {
        "en": "Spray Chlorantraniliprole or Cartap Hydrochloride.",
        "kn": "Chlorantraniliprole ಅಥವಾ Cartap Hydrochloride ಸಿಂಪಡಿಸಿ."
    },

    # -------------------------
    # LEAF MINER
    # -------------------------
    "leaf miner": {
        "en": "Use Abamectin 0.5ml/L or Spinosad.",
        "kn": "Abamectin 0.5ml/L ಅಥವಾ Spinosad ಬಳಸಿ."
    },

    # -------------------------
    # FRUIT FLIES
    # -------------------------
    "fruit fly": {
        "en": "Use methyl eugenol traps; spray protein bait with insecticide.",
        "kn": "Methyl eugenol ಟ್ರ್ಯಾಪ್ ಬಳಸಿ; ಇನ್ಸೆಕ್ಟಿಸೈಡ್ ಮಿಶ್ರಿತ ಪ್ರೋಟೀನ್ ಬೇಟ್ ಸಿಂಪಡಿಸಿ."
    },

    # -------------------------
    # FUNGAL DISEASES
    # -------------------------
    "powdery mildew": {
        "en": "Spray wettable sulphur 3g/L or Hexaconazole.",
        "kn": "ವೆಟೆಬಲ್ ಸಲ್ಫರ್ 3g/L ಅಥವಾ Hexaconazole ಸಿಂಪಡಿಸಿ."
    },

    "downy mildew": {
        "en": "Use Metalaxyl + Mancozeb 2g/L; improve air circulation.",
        "kn": "Metalaxyl + Mancozeb 2g/L ಬಳಸಿ; ಗಾಳಿ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ."
    },

    "blight": {
        "en": "Spray Mancozeb or Copper oxychloride 2g/L.",
        "kn": "Mancozeb ಅಥವಾ Copper oxychloride 2g/L ಸಿಂಪಡಿಸಿ."
    },

    "leaf spot": {
        "en": "Use Chlorothalonil or Mancozeb 2g/L.",
        "kn": "Chlorothalonil ಅಥವಾ Mancozeb 2g/L ಬಳಸಿ."
    },

    "anthracnose": {
        "en": "Spray Carbendazim or Propiconazole.",
        "kn": "Carbendazim ಅಥವಾ Propiconazole ಸಿಂಪಡಿಸಿ."
    },

    "root rot": {
        "en": "Apply Trichoderma around root zone; improve drainage.",
        "kn": "ರೂಟ್ ಜೋನ್‌ನಲ್ಲಿ Trichoderma ಬಳಸಿ; ನೀರು ನಿಲುಕದಂತೆ ನೋಡಿಕೊಳ್ಳಿ."
    },

    # -------------------------
    # BACTERIAL DISEASES
    # -------------------------
    "bacterial_wilt": {
        "en": "Drench soil with bleaching powder solution; avoid waterlogging.",
        "kn": "ಬ್ಲೀಚಿಂಗ್ ಪೌಡರ್ ದ್ರಾವಣ ಮಣ್ಣಿನಲ್ಲಿ ಹಾಕಿ; ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
    },

    "bacterial_blight": {
        "en": "Spray Copper oxychloride 2g/L; remove infected leaves.",
        "kn": "Copper oxychloride 2g/L ಸಿಂಪಡಿಸಿ; ಸೋಂಕಿತ ಎಲೆ ತೆಗೆದುಹಾಕಿ."
    },

    # -------------------------
    # VIRAL DISEASES
    # -------------------------
    "leaf curl": {
        "en": "Control whiteflies and thrips; remove infected plants; spray neem oil.",
        "kn": "ವೈಟ್‌ಫ್ಲೈ ಮತ್ತು ಥ್ರಿಪ್ಸ್ ತಡೆ ಮಾಡಿ; ಸೋಂಕಿತ ಗಿಡ ತೆಗೆಯಿರಿ; ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ."
    },

    "yellow mosaic": {
        "en": "Destroy infected plants; control vectors (whitefly); use neem oil.",
        "kn": "ಸೋಂಕಿತ ಗಿಡ ತೆಗೆದುಹಾಕಿ; ವೆಕ್ಟರ್ (ವೈಟ್‌ಫ್ಲೈ) ತಡೆ ಮಾಡಿ; ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ."
    },

    "mild mosaic": {
        "en": "Remove infected leaves; spray neem oil; improve nutrition.",
        "kn": "ಸೋಂಕಿತ ಎಲೆ ತೆಗೆಯಿರಿ; ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ; ಪೋಷಕಾಂಶ ಹೆಚ್ಚಿಸಿ."
    },

    # -------------------------
    # OTHER IMPORTANT PESTS
    # -------------------------
    "termite": {
        "en": "Apply Chlorpyrifos 2.5ml/L to soil; destroy mud galleries.",
        "kn": "Chlorpyrifos 2.5ml/L ಮಣ್ಣಿನಲ್ಲಿ ಬಳಸಿ; ಮಣ್ಣು ಮನೆಗಳನ್ನೂ ನಾಶಮಾಡಿ."
    },

    "weevil": {
        "en": "Use Imidacloprid soil drench; maintain field hygiene.",
        "kn": "Imidacloprid ಮಣ್ಣು ದ್ರಾವಕ ಮಾಡಿ; ಹೊಲ ಸ್ವಚ್ಛತೆ ಕಾಯ್ದುಕೊಳ್ಳಿ."
    },

    "nematode": {
        "en": "Apply neem cake 500kg/ha; use Trichoderma; solarize soil.",
        "kn": "500kg/ha ನೀಮ್ ಕೇಕ್ ನೀಡಿ; Trichoderma ಬಳಸಿ; ಮಣ್ಣು ಸೊಲರೈಸೇಶನ್ ಮಾಡಿ."
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

    # -------------------------
    # FOOD GRAINS
    # -------------------------
    "paddy": 6,
    "ragi": 4,
    "jowar": 5,
    "maize": 6,
    "bajra": 4,
    "wheat": 5,

    # -------------------------
    # PULSES
    # -------------------------
    "pigeon_pea": 4,
    "green_gram": 3.5,
    "black_gram": 3.5,
    "bengal_gram": 3,
    "horse_gram": 2.5,
    "cowpea": 3,

    # -------------------------
    # OILSEEDS
    # -------------------------
    "groundnut": 5,
    "sunflower": 6,
    "soybean": 4.5,
    "sesame": 3.5,
    "castor": 5,
    "safflower": 3.5,

    # -------------------------
    # COMMERCIAL CROPS
    # -------------------------
    "sugarcane": 7,
    "cotton": 5,
    "tobacco": 4.5,

    # -------------------------
    # PLANTATION CROPS
    # -------------------------
    "coffee": 5,
    "tea": 5.5,
    "arecanut": 6,
    "coconut": 7,
    "rubber": 5.5,
    "cashew": 4,

    # -------------------------
    # FRUITS
    # -------------------------
    "mango": 4.5,
    "banana": 7,
    "grapes": 5,
    "pomegranate": 4.5,
    "papaya": 5,
    "guava": 4,
    "sapota": 4.5,
    "orange": 5,

    # -------------------------
    # VEGETABLES
    # -------------------------
    "onion": 4,
    "tomato": 5,
    "potato": 5,
    "brinjal": 4.5,
    "cabbage": 5,
    "cauliflower": 4.5,
    "beans": 4,
    "cucumber": 4.5,

    # -------------------------
    # SPICES
    # -------------------------
    "turmeric": 5,
    "ginger": 5,
    "coriander": 3.5,
    "pepper": 4,
    "cardamom": 5.5
}


SOIL_WATER_HOLDING = {
    "sandy": 0.6,
    "loamy": 1.0,
    "clay": 1.3,

    # Extended:
    "sandy_loam": 0.8,
    "loam": 1.0,
    "clay_loam": 1.2,
    "silt_loam": 1.1,
    "red_soil": 0.9,
    "black_soil": 1.3,
    "laterite": 0.85
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

    # -------------------------
    # FOOD GRAINS – CEREALS
    # -------------------------
    "paddy": 5.0,
    "ragi": 2.0,
    "jowar": 1.8,
    "maize": 4.5,
    "bajra": 1.4,
    "wheat": 3.0,

    # -------------------------
    # PULSES
    # -------------------------
    "pigeon_pea": 1.0,       # Tur
    "green_gram": 0.8,       # Moong
    "black_gram": 0.7,       # Urad
    "bengal_gram": 1.0,      # Chickpea
    "horse_gram": 0.6,
    "cowpea": 0.8,

    # -------------------------
    # OILSEEDS
    # -------------------------
    "groundnut": 2.0,
    "sunflower": 1.5,
    "soybean": 1.8,
    "sesame": 0.5,
    "castor": 1.5,
    "safflower": 0.8,

    # -------------------------
    # COMMERCIAL / CASH CROPS
    # -------------------------
    "sugarcane": 80.0,       # tons cane per ha
    "cotton": 1.2,
    "tobacco": 1.6,

    # -------------------------
    # PLANTATION CROPS
    # -------------------------
    "coffee": 0.8,           # green beans
    "tea": 2.5,
    "arecanut": 2.5,
    "coconut": 10.0,         # tons of nuts equivalent
    "rubber": 1.5,           # dry rubber
    "cashew": 1.0,

    # -------------------------
    # FRUITS
    # -------------------------
    "mango": 8.0,
    "banana": 40.0,
    "grapes": 20.0,
    "pomegranate": 8.0,
    "papaya": 35.0,
    "guava": 12.0,
    "sapota": 10.0,
    "orange": 12.0,

    # -------------------------
    # VEGETABLES
    # -------------------------
    "onion": 20.0,
    "tomato": 25.0,
    "potato": 20.0,
    "brinjal": 30.0,
    "cabbage": 35.0,
    "cauliflower": 25.0,
    "beans": 10.0,
    "cucumber": 18.0,

    # -------------------------
    # SPICES
    # -------------------------
    "turmeric": 8.0,
    "ginger": 12.0,
    "coriander": 1.0,
    "pepper": 2.0,
    "cardamom": 1.5
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

    # -------------------------
    # FOOD GRAINS
    # -------------------------
    "paddy": [
        {"cond": "high_humidity", "disease": "blast"},
        {"cond": "continuous_rain", "disease": "bacterial blight"},
        {"cond": "rainy", "disease": "sheath blight"},
        {"cond": "high_temp_low_humidity", "disease": "brown spot"}
    ],

    "ragi": [
        {"cond": "high_humidity", "disease": "blast"},
        {"cond": "rainy", "disease": "leaf spot"}
    ],

    "jowar": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "high_temp", "disease": "sooty stripe"}
    ],

    "maize": [
        {"cond": "high_humidity", "disease": "turcicum leaf blight"},
        {"cond": "continuous_rain", "disease": "downy mildew"}
    ],

    "bajra": [
        {"cond": "high_humidity", "disease": "downy mildew"}
    ],

    "wheat": [
        {"cond": "low_humidity", "disease": "rust"},
        {"cond": "high_humidity", "disease": "leaf blight"}
    ],

    # -------------------------
    # PULSES
    # -------------------------
    "pigeon_pea": [
        {"cond": "high_humidity", "disease": "wilt"},
        {"cond": "rainy", "disease": "pod borer outbreak"}
    ],

    "green_gram": [
        {"cond": "high_humidity", "disease": "yellow mosaic virus"},
        {"cond": "hot_dry", "disease": "thrips outbreak"}
    ],

    "black_gram": [
        {"cond": "high_humidity", "disease": "leaf spot"},
        {"cond": "continuous_rain", "disease": "root rot"}
    ],

    "bengal_gram": [
        {"cond": "high_humidity", "disease": "blight"},
        {"cond": "rainy", "disease": "botrytis grey mold"}
    ],

    "horse_gram": [
        {"cond": "high_temp_low_humidity", "disease": "leaf spot"}
    ],

    "cowpea": [
        {"cond": "high_humidity", "disease": "anthracnose"},
        {"cond": "rainy", "disease": "root rot"}
    ],

    # -------------------------
    # OILSEEDS
    # -------------------------
    "groundnut": [
        {"cond": "high_humidity", "disease": "leaf spot"},
        {"cond": "continuous_rain", "disease": "root rot"},
        {"cond": "high_temp_low_humidity", "disease": "thrips"}
    ],

    "sunflower": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "rainy", "disease": "stem rot"}
    ],

    "soybean": [
        {"cond": "high_humidity", "disease": "rust"},
        {"cond": "continuous_rain", "disease": "root rot"}
    ],

    "sesame": [
        {"cond": "high_temp", "disease": "phyllody"},
        {"cond": "rainy", "disease": "leaf spot"}
    ],

    "castor": [
        {"cond": "high_humidity", "disease": "grey mold"},
        {"cond": "hot_dry", "disease": "jassids"}
    ],

    "safflower": [
        {"cond": "rainy", "disease": "rust"}
    ],

    # -------------------------
    # COMMERCIAL CROPS
    # -------------------------
    "sugarcane": [
        {"cond": "high_humidity", "disease": "red rot"},
        {"cond": "rainy", "disease": "smut"},
        {"cond": "high_temp", "disease": "borer outbreak"}
    ],

    "cotton": [
        {"cond": "high_humidity", "disease": "leaf spot"},
        {"cond": "hot_dry", "disease": "whitefly outbreak"},
        {"cond": "heavy_rain", "disease": "boll rot"}
    ],

    "tobacco": [
        {"cond": "high_humidity", "disease": "black shank"},
        {"cond": "hot_dry", "disease": "thrips"}
    ],

    # -------------------------
    # PLANTATION CROPS
    # -------------------------
    "coffee": [
        {"cond": "high_humidity", "disease": "rust"},
        {"cond": "continuous_rain", "disease": "berry borer outbreak"}
    ],

    "tea": [
        {"cond": "high_humidity", "disease": "blister blight"},
        {"cond": "rainy", "disease": "root rot"}
    ],

    "arecanut": [
        {"cond": "heavy_rain", "disease": "kole roga"},
        {"cond": "high_humidity", "disease": "bud rot"}
    ],

    "coconut": [
        {"cond": "high_humidity", "disease": "bud rot"},
        {"cond": "heavy_rain", "disease": "stem bleeding"}
    ],

    "rubber": [
        {"cond": "high_humidity", "disease": "powdery mildew"}
    ],

    "cashew": [
        {"cond": "high_humidity", "disease": "anthracnose"},
        {"cond": "rainy", "disease": "stem borer outbreak"}
    ],

    # -------------------------
    # FRUITS
    # -------------------------
    "mango": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "rainy", "disease": "anthracnose"}
    ],

    "banana": [
        {"cond": "high_humidity", "disease": "sigatoka"},
        {"cond": "continuous_rain", "disease": "panama wilt"}
    ],

    "grapes": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "rainy", "disease": "anthracnose"}
    ],

    "pomegranate": [
        {"cond": "high_temp_low_humidity", "disease": "fruit borer"},
        {"cond": "high_humidity", "disease": "bacterial blight"}
    ],

    "papaya": [
        {"cond": "high_temp_low_humidity", "disease": "mites"},
        {"cond": "high_humidity", "disease": "powdery mildew"}
    ],

    "guava": [
        {"cond": "high_humidity", "disease": "anthracnose"},
        {"cond": "rainy", "disease": "fruit fly outbreak"}
    ],

    "sapota": [
        {"cond": "high_humidity", "disease": "sooty mold"}
    ],

    "orange": [
        {"cond": "high_humidity", "disease": "citrus canker"},
        {"cond": "rainy", "disease": "root rot"}
    ],

    # -------------------------
    # VEGETABLES
    # -------------------------
    "onion": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "rainy", "disease": "purple blotch"}
    ],

    "tomato": [
        {"cond": "high_humidity", "disease": "late blight"},
        {"cond": "hot_dry", "disease": "leaf curl virus"},
        {"cond": "rainy", "disease": "bacterial wilt"}
    ],

    "potato": [
        {"cond": "high_humidity", "disease": "late blight"},
        {"cond": "rainy", "disease": "early blight"}
    ],

    "brinjal": [
        {"cond": "high_humidity", "disease": "phomopsis blight"},
        {"cond": "hot_dry", "disease": "mite outbreak"}
    ],

    "cabbage": [
        {"cond": "high_humidity", "disease": "black rot"},
        {"cond": "rainy", "disease": "downy mildew"}
    ],

    "cauliflower": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "rainy", "disease": "curd rot"}
    ],

    "beans": [
        {"cond": "high_humidity", "disease": "rust"},
        {"cond": "rainy", "disease": "anthracnose"}
    ],

    "cucumber": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "rainy", "disease": "downy mildew"}
    ],

    # -------------------------
    # SPICES
    # -------------------------
    "turmeric": [
        {"cond": "high_humidity", "disease": "leaf blotch"},
        {"cond": "continuous_rain", "disease": "rhizome rot"}
    ],

    "ginger": [
        {"cond": "high_humidity", "disease": "soft rot"},
        {"cond": "rainy", "disease": "leaf spot"}
    ],

    "coriander": [
        {"cond": "high_humidity", "disease": "powdery mildew"}
    ],

    "pepper": [
        {"cond": "heavy_rain", "disease": "quick wilt"},
        {"cond": "high_humidity", "disease": "anthracnose"}
    ],

    "cardamom": [
        {"cond": "high_humidity", "disease": "capsule rot"},
        {"cond": "continuous_rain", "disease": "rhizome rot"}
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

    # -------------------------
    # COLOR CHANGES
    # -------------------------
    "yellow leaves": ["nutrient deficiency", "water stress", "yellow mosaic virus", "root rot"],
    "yellowing between veins": ["iron deficiency", "zinc deficiency"],
    "reddish leaves": ["phosphorus deficiency", "viral infection"],
    "purple leaves": ["cold stress", "phosphorus deficiency"],

    # -------------------------
    # CURLING & DISTORTION
    # -------------------------
    "leaf curl": ["leaf curl virus", "whitefly", "thrips", "mites"],
    "upward curling leaves": ["potassium deficiency", "thrips"],
    "downward curling leaves": ["viral disease", "herbicide injury"],
    "leaf deformation": ["virus", "mites", "thrips"],

    # -------------------------
    # SPOTS
    # -------------------------
    "white spots": ["powdery mildew", "mites"],
    "brown spots": ["leaf spot", "blast", "blight"],
    "black spots": ["anthracnose", "fungal infection"],
    "yellow spots": ["fungal blight", "early blight"],
    "ring spots": ["viral infection"],
    "shot hole spots": ["bacterial leaf spot"],

    # -------------------------
    # PATCHES & MOULDS
    # -------------------------
    "white powder": ["powdery mildew"],
    "black mould": ["sooty mould", "aphid infestation"],
    "grey mould": ["botrytis", "fungal infection"],

    # -------------------------
    # WILTING
    # -------------------------
    "wilting": ["bacterial wilt", "root rot", "water stress"],
    "sudden drooping": ["stem borer", "wilt", "root damage"],
    "partial wilting": ["nematode", "root rot"],

    # -------------------------
    # HOLES & CHEWING DAMAGE
    # -------------------------
    "holes in leaves": ["leaf folder", "caterpillar", "beetles"],
    "chewed leaves": ["armyworm", "caterpillar"],
    "skeletonized leaves": ["leaf folder", "grasshopper"],
    "tunnels in leaves": ["leaf miner"],

    # -------------------------
    # FRUIT SYMPTOMS
    # -------------------------
    "fruit rot": ["anthracnose", "bacterial rot"],
    "fruit cracking": ["irregular irrigation", "boron deficiency"],
    "fruit borer holes": ["fruit borer"],
    "fruit drop": ["thrips", "mites", "nutrient deficiency", "heat stress"],
    "sticky fruits": ["whitefly", "mealybug"],

    # -------------------------
    # STEM & ROOT SYMPTOMS
    # -------------------------
    "stem rot": ["rhizoctonia", "fungal rot"],
    "stem borer holes": ["stem borer"],
    "root rot": ["phytophthora", "waterlogging"],
    "blackened roots": ["fungal wilt", "root rot"],

    # -------------------------
    # PEST SIGNS
    # -------------------------
    "tiny insects on leaves": ["aphid", "whitefly", "thrips"],
    "web on leaves": ["mites"],
    "sticky leaves": ["whitefly", "aphids"],
    "ant activity on plant": ["mealybug", "aphids"],

    # -------------------------
    # FUNGAL GROWTH SIGNS
    # -------------------------
    "white fungus": ["powdery mildew"],
    "grey fungus": ["downy mildew"],
    "black fungus": ["sooty mould"],

    # -------------------------
    # VIRAL SIGNS
    # -------------------------
    "mosaic pattern": ["yellow mosaic virus", "mosaic virus"],
    "vein clearing": ["viral disease"],
    "stunted growth": ["virus", "nematodes", "nutrient deficiency"],

    # -------------------------
    # DROUGHT / IRRIGATION SYMPTOMS
    # -------------------------
    "dry leaf tips": ["water stress", "potassium deficiency"],
    "leaf scorch": ["heat stress", "water shortage"],

    # -------------------------
    # WEATHER-RELATED SYMPTOMS
    # -------------------------
    "fungal patches after rain": ["anthracnose", "downy mildew"],
    "rotting after heavy rain": ["root rot", "stem rot"],

    # -------------------------
    # FRAGRANCE / SMELL SYMPTOMS
    # -------------------------
    "bad smell from roots": ["bacterial rot", "anaerobic soil"],

    # -------------------------
    # CROP-SPECIFIC ADDITIONS
    # -------------------------
    "neck blast": ["blast"],
    "sheath blotches": ["sheath blight"],
    "panicle drying": ["blast", "heat stress"],
    "fruit blackening": ["anthracnose", "sunburn"],
    "leaf blotch": ["fungal leaf spot"],

    # -------------------------
    # GENERAL NON-SPECIFIC SIGNS
    # -------------------------
    "slow growth": ["nutrient deficiency", "nematodes", "viral disease"],
    "poor flowering": ["micronutrient deficiency", "water stress"],
    "low fruit set": ["pollination issues", "pest damage"],
    "overall yellowing": ["nitrogen deficiency", "poor drainage"]
}


SYMPTOM_SYNONYMS = {

    # -------------------------
    # Yellowing Symptoms
    # -------------------------
    "yellowing": "yellow leaves",
    "leaves turning yellow": "yellow leaves",
    "yellow leaf": "yellow leaves",
    "pale leaves": "yellow leaves",
    "chlorosis": "yellow leaves",

    # -------------------------
    # Curling Symptoms
    # -------------------------
    "curling": "leaf curl",
    "curled leaves": "leaf curl",
    "twisted leaves": "leaf curl",
    "rolled leaves": "leaf curl",
    "leaf twisting": "leaf curl",

    # -------------------------
    # White Powder / Spots
    # -------------------------
    "white powder": "white spots",
    "powder on leaves": "white spots",
    "dusty white layer": "white spots",
    "powdery coating": "white spots",

    # -------------------------
    # Black Mould / Sooty
    # -------------------------
    "black powder": "black mould",
    "black layer": "black mould",
    "black stain": "black mould",
    "sooty layer": "black mould",
    "sticky black coating": "black mould",

    # -------------------------
    # Brown Spots
    # -------------------------
    "brown dots": "brown spots",
    "brown patches": "brown spots",
    "necrotic spots": "brown spots",
    "dry spots": "brown spots",

    # -------------------------
    # White / Grey Fungal Growth
    # -------------------------
    "white fungus": "white fungus",
    "white fungal growth": "white fungus",
    "grey fungus": "grey fungus",
    "mold on leaves": "grey fungus",

    # -------------------------
    # Wilting
    # -------------------------
    "drooping": "wilting",
    "plants falling": "wilting",
    "sudden wilt": "wilting",
    "weak plants": "wilting",

    # -------------------------
    # Chewing Damage
    # -------------------------
    "holes in leaves": "holes in leaves",
    "chewed leaves": "chewed leaves",
    "eaten leaves": "chewed leaves",
    "leaf eaten": "chewed leaves",

    # -------------------------
    # Leaf Miner
    # -------------------------
    "zigzag lines": "tunnels in leaves",
    "white trails": "tunnels in leaves",
    "scribbles on leaves": "tunnels in leaves",

    # -------------------------
    # Fruit Symptoms
    # -------------------------
    "fruit cracking": "fruit cracking",
    "fruit split": "fruit cracking",
    "fruit rot": "fruit rot",
    "black fruit": "fruit blackening",
    "fruit dropping": "fruit drop",
    "fruit fall": "fruit drop",

    # -------------------------
    # Sticky Honeydew
    # -------------------------
    "sticky leaves": "sticky leaves",
    "gum on leaves": "sticky leaves",
    "sticky layer": "sticky leaves",
    "sugar-like sticky": "sticky leaves",

    # -------------------------
    # Insect Presence
    # -------------------------
    "small insects": "tiny insects on leaves",
    "tiny insects": "tiny insects on leaves",
    "white insects": "whiteflies",
    "jumping insects": "thrips",

    # -------------------------
    # Virus Symptoms
    # -------------------------
    "mosaic": "mosaic pattern",
    "patchy color": "mosaic pattern",
    "vein clearing": "vein clearing",
    "stunted growth": "stunted growth",

    # -------------------------
    # Rot / Foul Smell
    # -------------------------
    "root smell": "bad smell from roots",
    "foul smell": "bad smell from roots",
    "root decay": "root rot",
    "rotting roots": "root rot",

    # -------------------------
    # Weather-Induced
    # -------------------------
    "burnt leaves": "leaf scorch",
    "scorching": "leaf scorch",
    "sunburn": "leaf scorch",
    "dry tips": "dry leaf tips",

    # -------------------------
    # Early/Farmer Slang
    # -------------------------
    "leaf burn": "leaf scorch",
    "leaf drying": "dry leaf tips",
    "weak crop": "slow growth",
    "growth stopped": "slow growth",
    "no flowering": "poor flowering",
    "low fruit set": "low fruit set",

    # -------------------------
    # Kannada Transliteration Handling
    # (Farmers often type symptoms phonetically in English)
    # -------------------------
    "haladi ele": "yellow leaves",      # “yellow leaf”
    "mullu kaddi": "leaf curl",         # “curled leaf/stem”
    "bili pudi": "white spots",         # “white powder”
    "kappu pudi": "black mould",        # “black powder”
    "chilu chilu hode": "holes in leaves",
    "hannina sirike": "fruit cracking"
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

    # ----------------------------------------------------
    # INCOME SUPPORT & DIRECT BENEFIT
    # ----------------------------------------------------
    "pm kisan": {
        "en": "PM-Kisan provides ₹6000/year to farmers in 3 installments. Eligibility: small & marginal farmers with valid land records.",
        "kn": "PM-Kisan ಯೋಜನೆ ರೈತರಿಗೆ ವರ್ಷಕ್ಕೆ ₹6000 (3 ಕಂತುಗಳಲ್ಲಿ) ನೀಡುತ್ತದೆ. ಅರ್ಹತೆ: ಭೂ ದಾಖಲೆ ಇರುವ ಸಣ್ಣ/ಅಂಚಿನ ರೈತರು."
    },

    "raitha shakti yojana": {
        "en": "Karnataka provides diesel subsidy of ₹250/acre for vehicle-based farm operations.",
        "kn": "ರೈತ ಶಕ್ತಿ ಯೋಜನೆ ಅಡಿಯಲ್ಲಿ ಕರ್ನಾಟಕ ಪ್ರತಿ ಎಕರೆ ₹250 ಡೀಸೆಲ್ ಸಹಾಯಧನ ನೀಡುತ್ತದೆ."
    },

    # ----------------------------------------------------
    # INSURANCE
    # ----------------------------------------------------
    "pmfby": {
        "en": "PMFBY provides crop insurance coverage against drought, flood, pests, diseases, and natural calamities.",
        "kn": "PMFBY ಬರ, ನೆರೆ, ಕೀಟ, ರೋಗ ಮತ್ತು ಪ್ರಕೃತಿ ವಿಕೋಪಗಳಿಂದ ಬೆಳೆ ನಷ್ಟಕ್ಕೆ ವಿಮೆ ಒದಗಿಸುತ್ತದೆ."
    },

    "karnataka crop insurance": {
        "en": "State crop insurance covers yield loss for paddy, maize, ragi, oilseeds & horticulture crops at subsidized premiums.",
        "kn": "ಕರ್ನಾಟಕ ಬೆಳೆ ವಿಮೆ ಯೋಜನೆ ಅಡಿಯಲ್ಲಿ ಅಕ್ಕಿ, ಜೋಳ, ರಾಗಿ, ಎಣ್ಣೆ ಬೀಜಗಳು ಮತ್ತು ತೋಟಗಾರಿಕೆ ಬೆಳೆಗಳಿಗೆ ವಿಮೆ ಲಭ್ಯ."
    },

    # ----------------------------------------------------
    # CREDIT, FINANCE & LOANS
    # ----------------------------------------------------
    "kcc": {
        "en": "Kisan Credit Card offers low-interest crop loans up to ₹3 lakhs with interest subvention.",
        "kn": "KCC ಕಡಿಮೆ ಬಡ್ಡಿದರದ ಬೆಳೆ ಸಾಲವನ್ನು ₹3 ಲಕ್ಷವರೆಗೆ ಒದಗಿಸುತ್ತದೆ."
    },

    "raitha siri": {
        "en": "Karnataka gives ₹10,000/ha per season for millet farmers (ragi, jowar, bajra).",
        "kn": "ರೈತ ಸಿರಿ ಯೋಜನೆ ಅಡಿಯಲ್ಲಿ ಮಿಲ್ಲೆಟ್ ರೈತರಿಗೆ ಪ್ರತಿ ಹೆಕ್ಟೇರ್ ₹10,000 ಸೀಸನ್‌ಗೂ ನೀಡಲಾಗುತ್ತದೆ."
    },

    # ----------------------------------------------------
    # IRRIGATION & WATER CONSERVATION
    # ----------------------------------------------------
    "drip subsidy": {
        "en": "Under PMKSY-Micro Irrigation, farmers get 55–75% subsidy for drip & sprinkler systems.",
        "kn": "PMKSY ಅಡಿಯಲ್ಲಿ ಡ್ರಿಪ್/ಸ್ಪ್ರಿಂಕ್ಲರ್ ನೀರಾವರಿಗೆ 55–75% ಸಹಾಯಧನ ಲಭ್ಯ."
    },

    "krishi bhagya": {
        "en": "Karnataka Krishi Bhagya supports polyhouse, shade net, farm ponds (Krishi Hondas), and micro irrigation.",
        "kn": "ಕೃಷಿ ಭಾಗ್ಯ ಯೋಜನೆ ಅಡಿಯಲ್ಲಿ ಪಾಲಿಹೌಸ್, ಶೇಡ್ ನೆಟ್, ಕೃಷಿ ಹೊಂಡ, ಮೈಕ್ರೋ ನೀರಾವರಿಗೆ ಸಹಾಯಧನ ನೀಡಲಾಗುತ್ತದೆ."
    },

    # ----------------------------------------------------
    # SOIL HEALTH & ORGANIC FARMING
    # ----------------------------------------------------
    "soil health card": {
        "en": "Provides soil nutrient status & fertilizer recommendation every 2 years.",
        "kn": "ಮಣ್ಣಿನ ಪೋಷಕಾಂಶ ಸ್ಥಿತಿ ಮತ್ತು ಗೊಬ್ಬರ ಸಲಹೆಯನ್ನು ಪ್ರತಿ 2 ವರ್ಷಕ್ಕೊಮ್ಮೆ ನೀಡುತ್ತದೆ."
    },

    "paramparagat krishi vikas yojana": {
        "en": "PKVY promotes organic farming with ₹50,000/ha for converting farmland into organic clusters.",
        "kn": "PKVY ಯೋಜನೆ ಪ್ರತಿ ಹೆಕ್ಟೇರ್ ₹50,000 ಸಹಾಯಧನದಿಂದ ಜೈವಿಕ ಕೃಷಿಯನ್ನು ಉತ್ತೇಜಿಸುತ್ತದೆ."
    },

    "organic farming policy karnataka": {
        "en": "Karnataka supports organic input subsidies and certification for organic growers.",
        "kn": "ಕರ್ನಾಟಕ ಜೈವಿಕ ಕೃಷಿಗೆ ಇನ್‌ಪುಟ್ ಸಹಾಯಧನ ಹಾಗೂ ಪ್ರಮಾಣೀಕರಣ ನೆರವನ್ನು ಒದಗಿಸುತ್ತದೆ."
    },

    # ----------------------------------------------------
    # FARM MACHINERY & EQUIPMENT SUBSIDY
    # ----------------------------------------------------
    "farm mechanization": {
        "en": "Government provides 40–60% subsidy on power tillers, tractors, sprayers, harvesters.",
        "kn": "ಸರಕಾರ 40–60% ಸಹಾಯಧನವನ್ನು ಪವರ್ ಟಿಲ್ಲರ್, ಟ್ರಾಕ್ಟರ್, ಸ್ಪ್ರೇಯರ್ ಮತ್ತು ಹಾರ್ವೆಸ್ಟರ್‌ಗಳಿಗೆ ನೀಡುತ್ತದೆ."
    },

    "custom hiring centers": {
        "en": "Subsidy for establishing CHCs to make farm machinery available at low rent.",
        "kn": "ಕೃಷಿ ಯಂತ್ರೋಪಕರಣಗಳನ್ನು ಕಡಿಮೆ ಬಾಡಿಗೆಯಲ್ಲಿ ಲಭ್ಯವಾಗುವಂತೆ CHC ಕೇಂದ್ರಗಳ ಸ್ಥಾಪನೆಗೆ ಸಹಾಯಧನ."
    },

    # ----------------------------------------------------
    # HORTICULTURE SCHEMES
    # ----------------------------------------------------
    "midh": {
        "en": "Mission for Integrated Development of Horticulture supports orchard establishment, nurseries, drip irrigation & cold storage.",
        "kn": "MIDH ಯೋಜನೆ ತೋಟಗಳ ಸ್ಥಾಪನೆ, ನರ್ಸರಿ, ಡ್ರಿಪ್ ನೀರಾವರಿ ಮತ್ತು ಕೋಲ್ಡ್ ಸ್ಟೋರೇಜ್‌ಗೆ ಬೆಂಬಲ ನೀಡುತ್ತದೆ."
    },

    "national horticulture mission": {
        "en": "50–60% subsidy for fruit crops (mango, banana, pomegranate, guava).",
        "kn": "ಮಾವಿನಹಣ್ಣು, ಬಾಳೆ, ದಾಳಿಂಬೆ, ಪೇರಳೆ ಮೊದಲಾದ ಹಣ್ಣು ತೋಟಗಳಿಗೆ 50–60% ಸಹಾಯಧನ."
    },

    # ----------------------------------------------------
    # LIVESTOCK & DAIRY
    # ----------------------------------------------------
    "nddp": {
        "en": "National Dairy Plan supports cattle breed improvement & milk productivity.",
        "kn": "ರಾಷ್ಟ್ರೀಯ ಹಾಲು ಯೋಜನೆ ಜಾತಿ ಸುಧಾರಣೆ ಮತ್ತು ಹಾಲು ಉತ್ಪಾದನೆಗೆ ಬೆಂಬಲ ನೀಡುತ್ತದೆ."
    },

    "poultry scheme": {
        "en": "Subsidy available for backyard poultry units & feed support.",
        "kn": "ಹಿಂಬಾಗಿಲಿನ ಕೋಳಿ ಸಾಕಾಣಿಕೆಗೆ ಸಹಾಯಧನ ಲಭ್ಯ."
    },

    # ----------------------------------------------------
    # STORAGE, MARKETS, MSP
    # ----------------------------------------------------
    "pm matsya sampada": {
        "en": "Support for fisheries, cold chain, storage & processing.",
        "kn": "ಮೀನುಗಾರಿಕೆ, ಕೋಲ್ಡ್ ಚೈನ್, ಸಂಗ್ರಹಣೆ ಮತ್ತು ಪ್ರಾಸೆಸಿಂಗ್‌ಗೆ ಸಹಾಯಧನ."
    },

    "msp procurement": {
        "en": "Government purchases crops at Minimum Support Price through APMC & FPOs.",
        "kn": "APMC/FPOಗಳ ಮೂಲಕ ಸರಕಾರ ಬೆಳೆಗಳನ್ನು MSP ದರದಲ್ಲಿ ಖರೀದಿಸುತ್ತದೆ."
    },

    "pmfme": {
        "en": "Provides 35% subsidy for food processing units under One District One Product (ODOP).",
        "kn": "ODOP ಅಡಿಯಲ್ಲಿ ಆಹಾರ ಪ್ರಾಸೆಸಿಂಗ್ ಘಟಕಗಳಿಗೆ 35% ಸಹಾಯಧನ."
    },

    # ----------------------------------------------------
    # FPO / FARMER PRODUCER ORGANIZATION SUPPORT
    # ----------------------------------------------------
    "fpo formation": {
        "en": "Govt supports forming FPOs with ₹18 lakh support over 3 years per FPO.",
        "kn": "ಪ್ರತಿ FPOಗೆ 3 ವರ್ಷಗಳಲ್ಲಿ ₹18 ಲಕ್ಷ ನೆರವು ನೀಡಿ ರೈತ ಉತ್ಪಾದಕ ಸಂಸ್ಥೆಗಳ ರಚನೆಗೆ ಬೆಂಬಲ."
    },

    "fpos under karnataka": {
        "en": "State supports market linkage, branding & post-harvest infrastructure for FPOs.",
        "kn": "ಕರ್ನಾಟಕದಲ್ಲಿ FPOಗಳಿಗೆ ಮಾರುಕಟ್ಟೆ ಸಂಪರ್ಕ, ಬ್ರಾಂಡಿಂಗ್ ಮತ್ತು ಕೊಯ್ಲಿನ ನಂತರದ ನೆಟ್‌ವರ್ಕ್‌ಗೆ ಸಹಾಯ."
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

    # ----------------------------------------------------
    # RED SOIL (Major soil of Karnataka)
    # ----------------------------------------------------
    "red soil": [
        "Groundnut", "Millet (Ragi, Jowar, Bajra)", "Pigeon pea", "Cotton",
        "Chilli", "Castor", "Horse gram", "Cowpea", "Sesame",
        "Tomato", "Onion", "Coriander", "Sunflower",
        "Mango", "Sapota", "Pomegranate"
    ],

    # ----------------------------------------------------
    # BLACK SOIL (Deep & medium black cotton soil)
    # ----------------------------------------------------
    "black soil": [
        "Cotton", "Soybean", "Paddy", "Red gram (Pigeon pea)", "Maize",
        "Sunflower", "Safflower", "Wheat", "Turmeric",
        "Onion", "Potato", "Sugarcane"
    ],

    # ----------------------------------------------------
    # LOAMY SOIL (Highly productive)
    # ----------------------------------------------------
    "loamy": [
        "Paddy", "Wheat", "Sugarcane", "Maize", "Vegetables",
        "Banana", "Papaya", "Tomato", "Cabbage", "Cauliflower", "Beans",
        "Grapes", "Orange", "Guava", "Turmeric", "Ginger"
    ],

    # ----------------------------------------------------
    # SANDY SOIL (Well drained, low fertility)
    # ----------------------------------------------------
    "sandy": [
        "Groundnut", "Watermelon", "Muskmelon", "Cucumber",
        "Onion", "Carrot", "Radish",
        "Cashew", "Coconut", "Arecanut",
        "Sesame", "Horse gram"
    ],

    # ----------------------------------------------------
    # CLAY SOIL (Moisture-retentive, nutrient rich)
    # ----------------------------------------------------
    "clay": [
        "Paddy", "Banana", "Sugarcane", "Vegetables",
        "Taro (Colocasia)", "Turmeric", "Ginger"
    ],

    # ----------------------------------------------------
    # LATERITE SOIL
    # ----------------------------------------------------
    "laterite": [
        "Arecanut", "Coconut", "Cashew", "Rubber", "Pepper", "Coffee",
        "Pineapple", "Banana", "Tapioca"
    ],

    # ----------------------------------------------------
    # ALLUVIAL SOIL (River basins)
    # ----------------------------------------------------
    "alluvial": [
        "Paddy", "Sugarcane", "Banana",
        "Potato", "Tomato", "Cabbage", "Onion", "Chilli",
        "Maize", "Wheat", "Green gram", "Black gram"
    ],

    # ----------------------------------------------------
    # COASTAL SALINE SOIL
    # ----------------------------------------------------
    "saline soil": [
        "Coconut", "Cashew", "Arecanut",
        "Paddy (salt-tolerant varieties)", "Fish–paddy integrated farming"
    ],

    # ----------------------------------------------------
    # FOREST / HILLY SOILS (Western Ghats)
    # ----------------------------------------------------
    "forest soil": [
        "Coffee", "Pepper", "Cardamom", "Arecanut",
        "Fruit crops like Banana, Pineapple",
        "Ginger", "Turmeric"
    ],

    # ----------------------------------------------------
    # SILTY SOIL
    # ----------------------------------------------------
    "silty": [
        "Paddy", "Vegetables", "Mustard", "Maize",
        "Groundnut", "Sunflower"
    ]
}


CLIMATE_TO_CROP = {

    # ----------------------------------------------------
    # 1. ARID / DRY (Low rainfall, high temperature)
    # Ballari, Raichur, Vijayapura regions
    # ----------------------------------------------------
    "dry": [
        "Ragi", "Jowar", "Bajra", "Foxtail Millet",
        "Castor", "Pigeon pea", "Sesame",
        "Horse gram", "Cowpea",
        "Guava", "Ber fruit",
        "Cactus fodder"
    ],

    # ----------------------------------------------------
    # 2. SEMI-ARID (Moderate rainfall 500–750 mm)
    # Tumkur, Chitradurga, Koppal etc.
    # ----------------------------------------------------
    "semi-dry": [
        "Cotton", "Groundnut", "Bengal gram",
        "Red gram (Pigeon pea)", "Sunflower",
        "Safflower", "Jowar",
        "Onion", "Chickpea",
        "Watermelon", "Muskmelon"
    ],

    # ----------------------------------------------------
    # 3. SEMI-HUMID (800–1200 mm rainfall)
    # Hassan, Chikkamagaluru plains, Dharwad belt
    # ----------------------------------------------------
    "semi-humid": [
        "Maize", "Soybean", "Vegetables",
        "Turmeric", "Ginger",
        "Sugarcane", "Paddy (rainfed)",
        "Banana", "Papaya",
        "Tomato", "Beans",
        "Mango", "Sapota"
    ],

    # ----------------------------------------------------
    # 4. HUMID / HIGH HUMIDITY (Western Ghats belt)
    # ----------------------------------------------------
    "humid": [
        "Paddy (irrigated)", "Banana", "Arecanut",
        "Black pepper", "Cardamom",
        "Ginger", "Turmeric",
        "Coconut", "Rubber",
        "Pineapple", "Tapioca"
    ],

    # ----------------------------------------------------
    # 5. COASTAL-HUMID (High rainfall, saline influence)
    # Coastal Karnataka: Udupi, Mangalore, Karwar
    # ----------------------------------------------------
    "coastal": [
        "Coconut", "Arecanut", "Paddy",
        "Cashew", "Pepper",
        "Banana", "Nutmeg",
        "Betel leaf",
        "Fish–paddy integrated farming"
    ],

    # ----------------------------------------------------
    # 6. HILL / HIGH ALTITUDE (Cool climate)
    # Kodagu, Chikkamagaluru, Nilgiri-like zones
    # ----------------------------------------------------
    "hill": [
        "Coffee", "Pepper", "Cardamom",
        "Orange", "Avocado",
        "Tea", "Cabbage", "Cauliflower",
        "Carrot", "Beetroot",
        "Ginger", "Pineapple"
    ],

    # ----------------------------------------------------
    # 7. SUB-TROPICAL (Mixed climate, moderate winters)
    # ----------------------------------------------------
    "subtropical": [
        "Wheat", "Mustard", "Barley",
        "Potato", "Pea",
        "Tomato", "Onion",
        "Sugarcane", "Paddy",
        "Chickpea", "Lentil"
    ],

    # ----------------------------------------------------
    # 8. HEAVY RAINFALL ZONE (2000–5000 mm)
    # Western Ghats upper slopes
    # ----------------------------------------------------
    "heavy_rainfall": [
        "Tea", "Coffee", "Arecanut",
        "Black pepper", "Ginger",
        "Paddy (traditional varieties)",
        "Jackfruit", "Breadfruit"
    ],

    # ----------------------------------------------------
    # 9. TEMPERATE COOL ZONES
    # (rare in Karnataka but useful for general AI support)
    # ----------------------------------------------------
    "temperate": [
        "Apple", "Pear", "Plum",
        "Cabbage", "Broccoli",
        "Carrot", "Lettuce",
        "Barley", "Oats"
    ]
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

    # -----------------------------
    # FOOD GRAINS (Cereals & Millets)
    # -----------------------------
    "paddy": 45000,
    "ragi": 30000,
    "jowar": 28000,
    "bajra": 25000,
    "maize": 35000,
    "wheat": 40000,
    "foxtail millet": 26000,
    "little millet": 25000,

    # -----------------------------
    # PULSES
    # -----------------------------
    "pigeon pea": 38000,
    "green gram": 30000,
    "black gram": 29000,
    "bengal gram": 32000,
    "horse gram": 20000,
    "cowpea": 24000,

    # -----------------------------
    # OILSEEDS
    # -----------------------------
    "groundnut": 50000,
    "sunflower": 42000,
    "soybean": 38000,
    "sesame": 28000,
    "castor": 30000,
    "safflower": 27000,

    # -----------------------------
    # COMMERCIAL / CASH CROPS
    # -----------------------------
    "sugarcane": 145000,
    "cotton": 65000,
    "tobacco": 80000,

    # -----------------------------
    # PLANTATION CROPS
    # -----------------------------
    "coffee": 160000,
    "tea": 180000,
    "arecanut": 130000,
    "coconut": 70000,
    "rubber": 120000,
    "cashew": 60000,

    # -----------------------------
    # FRUITS
    # -----------------------------
    "banana": 120000,
    "mango": 70000,
    "grapes": 150000,
    "pomegranate": 130000,
    "papaya": 90000,
    "guava": 65000,
    "sapota": 62000,
    "orange": 85000,

    # -----------------------------
    # VEGETABLES (Per hectare intensive)
    # -----------------------------
    "onion": 85000,
    "tomato": 90000,
    "potato": 110000,
    "brinjal": 75000,
    "cabbage": 65000,
    "cauliflower": 70000,
    "beans": 60000,
    "cucumber": 55000,

    # -----------------------------
    # SPICES & CONDIMENTS
    # -----------------------------
    "chilli": 80000,
    "turmeric": 95000,
    "ginger": 140000,
    "coriander": 45000,
    "pepper": 100000,
    "cardamom": 180000
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
MARKET_PRICE = {

    # -----------------------------
    # FOOD GRAINS (Cereals & Millets)
    # -----------------------------
    "paddy": 20,
    "ragi": 25,
    "jowar": 22,
    "bajra": 20,
    "maize": 20,
    "wheat": 24,
    "foxtail millet": 50,
    "little millet": 45,

    # -----------------------------
    # PULSES
    # -----------------------------
    "pigeon pea": 90,
    "green gram": 110,
    "black gram": 85,
    "bengal gram": 70,
    "horse gram": 50,
    "cowpea": 55,

    # -----------------------------
    # OILSEEDS
    # -----------------------------
    "groundnut": 50,
    "sunflower": 55,
    "soybean": 40,
    "sesame": 120,
    "castor": 48,
    "safflower": 45,

    # -----------------------------
    # COMMERCIAL / CASH CROPS
    # -----------------------------
    "sugarcane": 3,     # per kg (₹3000/ton)
    "cotton": 60,       # seed cotton
    "tobacco": 160,     # FCV tobacco auction price

    # -----------------------------
    # PLANTATION CROPS
    # -----------------------------
    "coffee": 180,      # cherry
    "tea": 120,
    "arecanut": 450,
    "coconut": 12,      # per nut equivalent
    "rubber": 150,
    "cashew": 120,

    # -----------------------------
    # FRUITS
    # -----------------------------
    "banana": 10,
    "mango": 30,
    "grapes": 25,
    "pomegranate": 80,
    "papaya": 12,
    "guava": 25,
    "sapota": 20,
    "orange": 20,

    # -----------------------------
    # VEGETABLES
    # -----------------------------
    "onion": 20,
    "tomato": 15,
    "potato": 20,
    "brinjal": 18,
    "cabbage": 12,
    "cauliflower": 15,
    "beans": 35,
    "cucumber": 12,

    # -----------------------------
    # SPICES & CONDIMENTS
    # -----------------------------
    "chilli": 70,
    "turmeric": 100,
    "ginger": 60,
    "coriander": 80,
    "pepper": 500,
    "cardamom": 1200
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



