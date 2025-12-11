# =========================================================
# main.py — KS Chatbot Backend (FastAPI + Gemini + Firebase)
# Extended with:
# - Stage-wise recommendations
# - Fertilizer calculator per stage
# - Pesticide recommendation engine
# - Irrigation schedule module
# - Yield prediction (simple heuristic)
# - Weather + crop-stage fusion advisory
# =========================================================

import os
import json
import requests
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re
import difflib
from collections import defaultdict, Counter
from typing import Tuple
from fastapi.staticfiles import StaticFiles
# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

if not FIREBASE_DATABASE_URL:
    raise Exception("FIREBASE_DATABASE_URL missing")

# -----------------------------
# Globals
# -----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]

credentials = None
client = None
active_chats: Dict[str, Any] = {}

app = FastAPI(title="KS Chatbot Backend", version="4.0")
app.mount("/tts", StaticFiles(directory="tts_audio"), name="tts")


# =========================================================
# MODELS
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

def generate_tts_audio(text: str, lang: str):
    # You can use Google TTS, gTTS or Gemini TTS (if enabled)
    from gtts import gTTS
    import uuid

    filename = f"tts_{uuid.uuid4()}.mp3"
    filepath = f"./tts_audio/{filename}"

    try:
        tts = gTTS(text=text, lang="kn" if lang == "kn" else "en")
        tts.save(filepath)
        return f"/tts/{filename}"
    except Exception as e:
        print("TTS error:", e)
        return None

# =========================================================
# Initialization helpers (Gemini + Firebase)
# =========================================================
def initialize_gemini():
    global client
    try:
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            print("Gemini initialized.")
        else:
            print("GEMINI_API_KEY not set; Gemini disabled.")
    except Exception as e:
        print("Gemini init error:", e)


def initialize_firebase_credentials():
    global credentials
    if credentials:
        return
    try:
        info = json.loads(SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        print("Firebase service account loaded.")
    except Exception as e:
        print("Cannot load Firebase credentials:", e)
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
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {e}")


# =========================================================
# Firebase helper
# =========================================================
def firebase_get(path: str):
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_DATABASE_URL}/{path}.json"
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Firebase GET error:", e)
        return None


# convenience fetchers
def get_language(user_id: str) -> str:
    lang = firebase_get(f"Users/{user_id}/preferredLanguage")
    if isinstance(lang, str) and lang.lower() == "kn":
        return "kn"
    return "en"


def get_user_farm_details(user_id: str) -> Dict[str, Any]:
    data = firebase_get(f"Users/{user_id}/farmDetails")
    return data if isinstance(data, dict) else {}


# =========================================================
# Existing modules: Soil center, weather placeholder, market, pest/disease, farm timeline
# (kept concise; unchanged from earlier versions)
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


PRICE_LIST = {
    "chilli": 50, "paddy": 20, "ragi": 18, "areca": 470,
    "banana": 12, "turmeric": 120, "cotton": 40, "sugarcane": 3
}


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


# =========================================================
# Stage-wise recommendation engine (existing)
# =========================================================
STAGE_RECOMMENDATIONS = {

    # =====================================================
    # 1. PADDY
    # =====================================================
    "paddy": {
        "nursery": {
            "en": "Maintain 2–3 cm water level; protect seedlings from pests.",
            "kn": "2–3 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಸಸಿಗಳನ್ನು ಕೀಟಗಳಿಂದ ರಕ್ಷಿಸಿ."
        },
        "tillering": {
            "en": "Apply urea (N); maintain 3–5 cm water; manage weeds.",
            "kn": "ಯೂರಿಯಾ (N) ನೀಡಿ; 3–5 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಗಿಡ್ಮುಳ್ಳು ನಿಯಂತ್ರಿಸಿ."
        },
        "panicle initiation": {
            "en": "Apply potash + micronutrients; ensure water flow.",
            "kn": "ಪೊಟಾಶ್ + ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ನೀರಾವರಿ ಸರಿಯಾಗಿ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Avoid irrigation for 5 days; protect from pests (BPH).",
            "kn": "5 ದಿನ ನೀರಾವರಿ ತಪ್ಪಿಸಿ; ಕೀಟ (BPH) ದಾಳಿಯಿಂದ ರಕ್ಷಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when 80% grains turn golden yellow.",
            "kn": "80% ಧಾನ್ಯ ಬಂಗಾರದ ಬಣ್ಣವಾಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 2. RAGI (Finger Millet)
    # =====================================================
    "ragi": {
        "germination": {
            "en": "Light irrigation; protect seedlings from early pests.",
            "kn": "ಹಗುರ ನೀರಾವರಿ ಮಾಡಿ; ಕೀಟಗಳಿಂದ ಸಸಿಗಳನ್ನು ರಕ್ಷಿಸಿ."
        },
        "tillering": {
            "en": "Apply NPK; weed control essential.",
            "kn": "NPK ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ನಿಯಂತ್ರಣೆ ಅಗತ್ಯ."
        },
        "flowering": {
            "en": "Maintain moisture; avoid stress.",
            "kn": "ಮಣ್ಣು ತೇವ ಕಾಪಾಡಿ; ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "grain filling": {
            "en": "Light irrigation; avoid lodging.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ಗಿಡ ಬಿದ್ದು ಹೋಗುವುದನ್ನು ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when earheads turn brown.",
            "kn": "ಕೋಲುಗಳು ಕಂದು ಬಣ್ಣ ಪಡೆದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 3. MAIZE
    # =====================================================
    "maize": {
        "vegetative": {
            "en": "Apply nitrogen; maintain soil moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಣ್ಣು ತೇವ ಕಾಪಾಡಿ."
        },
        "tasseling": {
            "en": "Irrigate heavily; avoid moisture stress.",
            "kn": "ಹೆಚ್ಚು ನೀರಾವರಿ ಮಾಡಿ; ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "silking": {
            "en": "Critical stage; maintain uniform moisture.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ಸಮಾನ ತೇವಾವಸ್ಥೆ ಇರಲಿ."
        },
        "grain filling": {
            "en": "Apply potash for proper grain development.",
            "kn": "ಧಾನ್ಯ ಬೆಳವಣಿಗೆಗೆ ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest when husk turns yellow & dry.",
            "kn": "ಹಸ್ಕ್ ಹಳದಿ/ಒಣಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 4. JOWAR (Sorghum)
    # =====================================================
    "jowar": {
        "vegetative": {
            "en": "Apply NPK; light irrigation.",
            "kn": "NPK ನೀಡಿ; ಹಗುರ ನೀರಾವರಿ."
        },
        "booting": {
            "en": "Irrigate; maintain weed-free field.",
            "kn": "ನೀರಾವರಿ ಮಾಡಿ; ಗಿಡ್ಮುಳ್ಳು ದೂರವಿಡಿ."
        },
        "flowering": {
            "en": "Critical moisture stage; avoid drought.",
            "kn": "ಮುಖ್ಯ ಹಂತ; ಬರ ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "grain filling": {
            "en": "Light irrigation; apply potash.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest when grains become hard.",
            "kn": "ಧಾನ್ಯ ಗಟ್ಟಿ ಆಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 5. TUR (Red Gram / Pigeon Pea)
    # =====================================================
    "tur": {
        "vegetative": {
            "en": "Apply nitrogen; ensure good sunlight.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಉತ್ತಮ ಸೂರ್ಯನ ಬೆಳಕು ಇರಲಿ."
        },
        "flowering": {
            "en": "Avoid waterlogging; control pod borer.",
            "kn": "ನೀರಿನ ನಿಲ್ಲಿಕೆ ತಪ್ಪಿಸಿ; ಪಾಡ್ ಬೋರರ್ ನಿಯಂತ್ರಿಸಿ."
        },
        "pod formation": {
            "en": "Spray micronutrients; maintain moisture.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ಸಿಂಪಡಿಸಿ; ತೇವ ಕಾಪಾಡಿ."
        },
        "maturity": {
            "en": "Harvest when pods dry & turn brown.",
            "kn": "ಪಾಡ್‌ಗಳು ಒಣಗಿ ಕಂದು ಬಣ್ಣ ಬಂದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 6. GREEN GRAM (Moong)
    # =====================================================
    "moong": {
        "vegetative": {
            "en": "Apply DAP; ensure weed-free field.",
            "kn": "DAP ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ನಿಯಂತ್ರಿಸಿ."
        },
        "flowering": {
            "en": "Light irrigation; avoid heavy rain.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ಹೆಚ್ಚು ಮಳೆಯಿದ್ದರೆ ತಪ್ಪಿಸಿ."
        },
        "pod setting": {
            "en": "Micronutrient spray; control sucking pests.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ಸ್ಯಕ್ಕಿಂಗ್ ಕೀಟ ನಿಯಂತ್ರಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when 80% pods mature.",
            "kn": "80% ಪಾಡ್‌ಗಳು ಹಸಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 7. BLACK GRAM (Urad)
    # =====================================================
    "urad": {
        "vegetative": {
            "en": "Early urea application; remove weeds.",
            "kn": "ಪ್ರಾರಂಭಿಕ ಯೂರಿಯಾ ಅನ್ವಯಿಸಿ; ಗಿಡ್ಮುಳ್ಳು ತೆಗೆದುಹಾಕಿ."
        },
        "flowering": {
            "en": "Light irrigation; protect from whitefly.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ವೈಟ್‌ಫ್ಲೈಯಿಂದ ರಕ್ಷಿಸಿ."
        },
        "pod setting": {
            "en": "Spray micronutrients.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ಸಿಂಪಡಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when pods turn black.",
            "kn": "ಪಾಡ್‌ಗಳು ಕಪ್ಪಾಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 8. GROUNDNUT
    # =====================================================
    "groundnut": {
        "vegetative": {
            "en": "Apply gypsum; maintain moisture.",
            "kn": "ಜಿಪ್ಸಮ್ ನೀಡಿ; ಮಣ್ಣು ತೇವ ಇರಲಿ."
        },
        "flowering": {
            "en": "Critical pegging stage; avoid dry soil.",
            "kn": "ಮುಖ್ಯ ಪೆಗ್ಗಿಂಗ್ ಹಂತ; ಒಣಮಣ್ಣು ತಪ್ಪಿಸಿ."
        },
        "pod development": {
            "en": "Apply calcium; light irrigation.",
            "kn": "ಕ್ಯಾಲ್ಸಿಯಂ ನೀಡಿ; ಹಗುರ ನೀರಾವರಿ."
        },
        "harvest": {
            "en": "Harvest when leaves turn yellow.",
            "kn": "ಎಲೆಗಳು ಹಳದಿಯಾಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 9. SUNFLOWER
    # =====================================================
    "sunflower": {
        "vegetative": {
            "en": "Apply NPK; maintain uniform spacing.",
            "kn": "NPK ನೀಡಿ; ಸಮಾನ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "bud formation": {
            "en": "Light irrigation; avoid waterlogging.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ನೀರಿನ ನಿಲ್ಲಿಕೆ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Micronutrient spray (boron).",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ (ಬೋರಾನ್) ಸಿಂಪಡಿಸಿ."
        },
        "seed filling": {
            "en": "Maintain moisture; protect from birds.",
            "kn": "ತೇವ ಕಾಪಾಡಿ; ಪಕ್ಷಿಗಳಿಂದ ರಕ್ಷಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when head turns brown.",
            "kn": "ಹೆಡ್ ಕಂದು ಬಣ್ಣ ಬಂದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 10. SESAME (Til)
    # =====================================================
    "sesame": {
        "vegetative": {
            "en": "Apply nitrogen; weed regularly.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ತೆಗೆದುಹಾಕಿ."
        },
        "flowering": {
            "en": "Light irrigation; avoid stress.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "capsule setting": {
            "en": "Apply micronutrients.",
            "kn": "ಮೈಕ್ರೋನ್ಯೂಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest when leaves drop & capsules dry.",
            "kn": "ಎಲೆಗಳು ಬೀಳಿದಾಗ ಮತ್ತು ಕ್ಯಾಪ್ಸುಲ್ ಒಣಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 11. SUGARCANE
    # =====================================================
    "sugarcane": {
        "tillering": {
            "en": "Apply nitrogen; maintain moisture.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ತೇವ ಕಾಪಾಡಿ."
        },
        "grand growth": {
            "en": "Irrigate frequently; apply potash.",
            "kn": "ನಿಯಮಿತ ನೀರಾವರಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "ripening": {
            "en": "Reduce irrigation; avoid lodging.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಗಿಡ ಬಿದ್ದುಹೋಗುವುದನ್ನು ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest 12–14 months after planting.",
            "kn": "ನೆಡುವ 12–14 ತಿಂಗಳ ನಂತರ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 12. COTTON
    # =====================================================
    "cotton": {
        "vegetative": {
            "en": "Apply nitrogen; maintain spacing.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಸಮಾನ ಅಂತರ ಕಾಯ್ದುಕೊಳ್ಳಿ."
        },
        "square formation": {
            "en": "Micronutrient spray; avoid leaf curl.",
            "kn": "ಮೈಕ್ರೋನ್ಯೂಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ಎಲೆ ಕರ್ಭಟ ತಪ್ಪಿಸಿ."
        },
        "flowering": {
            "en": "Irrigate regularly; manage bollworms.",
            "kn": "ನಿಯಮಿತ ನೀರಾವರಿ; ಬೋಲ್‌ವರ್ಮ್ ನಿಯಂತ್ರಿಸಿ."
        },
        "boll development": {
            "en": "Apply potash; keep field clean.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ಹೊಲ ಸ್ವಚ್ಛವಾಗಿಡಿ."
        },
        "harvest": {
            "en": "Pick cotton when bolls open fully.",
            "kn": "ಬೋಲ್ ಪೂರ್ಣತೆ ಪಡೆದಾಗ ಕೊರೆಯಿರಿ."
        }
    },

    # =====================================================
    # 13. ARECANUT
    # =====================================================
    "arecanut": {
        "vegetative": {
            "en": "Apply FYM + NPK; maintain moisture.",
            "kn": "FYM + NPK ನೀಡಿ; ತೇವ ಕಾಪಾಡಿ."
        },
        "flowering": {
            "en": "Spray boron; prevent drought stress.",
            "kn": "ಬೋರಾನ್ ನೀಡಿ; ಬರ ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "nut development": {
            "en": "Irrigate weekly; apply potash.",
            "kn": "ವಾರಕ್ಕೆ ನೀರಾವರಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest when nuts mature.",
            "kn": "ಕಾಯುಗಳು ಹಸಿದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 14. COCONUT
    # =====================================================
    "coconut": {
        "vegetative": {
            "en": "Apply FYM; irrigation essential.",
            "kn": "FYM ನೀಡಿ; ನೀರಾವರಿ ಅಗತ್ಯ."
        },
        "flowering": {
            "en": "Apply boron; remove weeds.",
            "kn": "ಬೋರಾನ್ ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ತೆಗೆದುಹಾಕಿ."
        },
        "nut formation": {
            "en": "Regular irrigation; apply potash.",
            "kn": "ನೀತಿಯ ನೀರಾವರಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest every 45–60 days based on maturity.",
            "kn": "ಹಸುವನ್ನು ಗಮನಿಸಿ ಪ್ರತಿ 45–60 ದಿನಕ್ಕೆ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 15. BANANA
    # =====================================================
    "banana": {
        "vegetative": {
            "en": "NPK application; remove suckers.",
            "kn": "NPK ನೀಡಿ; ಬದಿಯ ಸಕ್ಕರ್ಸ್ ತೆಗೆದುಹಾಕಿ."
        },
        "flowering": {
            "en": "Apply micronutrients; tie bunch.",
            "kn": "ಮೈಕ್ರೋನ್ಯೂಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ಗುಚ್ಛ ಕಟ್ಟಿ."
        },
        "fruiting": {
            "en": "Apply potash; maintain irrigation.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಾವರಿ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Harvest when fingers are well developed.",
            "kn": "ಬೆರಳುಗಳು ಗಟ್ಟಿ ಬೆಳವಣಿಗೆ ಪಡೆದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 16. MANGO
    # =====================================================
    "mango": {
        "vegetative": {
            "en": "Prune branches; apply FYM + micronutrients.",
            "kn": "ಕೊಂಬೆ ಕತ್ತರಿಸಿ; FYM ಮತ್ತು ಮೈಕ್ರೋನ್ಯೂಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ."
        },
        "flowering": {
            "en": "Light irrigation; avoid nitrogen.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ."
        },
        "fruit set": {
            "en": "Spray micronutrients; avoid moisture stress.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ತೇವ ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "maturity": {
            "en": "Harvest based on variety maturity index.",
            "kn": "ಪ್ರಭೇದದ ಪಕ್ವತೆಯ ಸೂಚಕದ ಆಧಾರದಲ್ಲಿ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 17. SAPOTA (Chikoo)
    # =====================================================
    "sapota": {
        "vegetative": {
            "en": "Apply manure + NPK.",
            "kn": "ಗೊಬ್ಬರ + NPK ನೀಡಿ."
        },
        "flowering": {
            "en": "Light irrigation; avoid pruning.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ಕತ್ತರಿಕೆ ಬೇಡ."
        },
        "fruiting": {
            "en": "Maintain moisture; apply potash.",
            "kn": "ತೇವ ಕಾಪಾಡಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest when fruits soften slightly.",
            "kn": "ಹಣ್ಣು ಸ್ವಲ್ಪ ಮೃದುವಾದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 18. GRAPES
    # =====================================================
    "grapes": {
        "pruning": {
            "en": "Apply FYM; prune properly for canopy.",
            "kn": "FYM ನೀಡಿ; canopyಗಾಗಿ ಸರಿಯಾದ ಕತ್ತರಿಕೆ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Avoid excess irrigation; spray micronutrients.",
            "kn": "ಅತಿಯಾದ ನೀರಾವರಿ ತಪ್ಪಿಸಿ; ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ."
        },
        "fruiting": {
            "en": "Potash application; protect from powdery mildew.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ನಿಯಂತ್ರಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when berries reach sugar content.",
            "kn": "ಹಣ್ಣು ಸಕ್ಕರೆಯ ಮಟ್ಟ ತಲುಪಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 19. TOMATO
    # =====================================================
    "tomato": {
        "vegetative": {
            "en": "Apply NPK; support staking.",
            "kn": "NPK ನೀಡಿ; ಸಟಿಂಗ್ ಮಾಡಿ."
        },
        "flowering": {
            "en": "Spray boron; maintain irrigation.",
            "kn": "ಬೋರಾನ್ ಸಿಂಪಡಿಸಿ; ನೀರಾವರಿ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply potash; control fruit borer.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ಫ್ರೂಟ್ ಬೋರರ್ ನಿಯಂತ್ರಿಸಿ."
        },
        "harvest": {
            "en": "Harvest at breaker stage.",
            "kn": "ಬ್ರೇಕರ್ ಹಂತದಲ್ಲಿ ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 20. BRINJAL
    # =====================================================
    "brinjal": {
        "vegetative": {
            "en": "Apply nitrogen; remove weeds.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ತೆಗೆದುಹಾಕಿ."
        },
        "flowering": {
            "en": "Micronutrient spray; avoid water stress.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ನೀರಾವರಿ ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "fruiting": {
            "en": "Control shoot & fruit borer.",
            "kn": "ಶೂಟ್ & ಫ್ರೂಟ್ ಬೋರರ್ ನಿಯಂತ್ರಿಸಿ."
        },
        "harvest": {
            "en": "Harvest tender fruits.",
            "kn": "ಮೃದುವಾದ ಹಣ್ಣುಗಳು ಬಂದಾಗ ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 21. ONION
    # =====================================================
    "onion": {
        "vegetative": {
            "en": "Apply nitrogen split dose; maintain moisture.",
            "kn": "ವಿಭಜಿತ ನೈಟ್ರೋಜನ್ ನೀಡಿ; ತೇವ ಇರಲಿ."
        },
        "bulb formation": {
            "en": "Apply potash; ensure irrigation.",
            "kn": "ಪೊಟಾಶ் ನೀಡಿ; ನೀರಾವರಿ ಇರಲಿ."
        },
        "maturation": {
            "en": "Stop irrigation before 10–15 days of harvest.",
            "kn": "ಕೊಯ್ತಿಗೆ 10–15 ದಿನ ಮೊದಲು ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when tops fall over.",
            "kn": "ಎಲೆಗಳು ಬಿದ್ದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 22. POTATO
    # =====================================================
    "potato": {
        "vegetative": {
            "en": "Earthing up required; apply NPK.",
            "kn": "ಎರ್ಥಿಂಗ್ ಅಪ್ ಮಾಡಿ; NPK ನೀಡಿ."
        },
        "tuber initiation": {
            "en": "Maintain moisture; avoid high temperature.",
            "kn": "ತೇವ ಇರಲಿ; ಹೆಚ್ಚು ಬಿಸಿಲು ತಪ್ಪಿಸಿ."
        },
        "bulking": {
            "en": "Apply potash; irrigate regularly.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನಿಯಮಿತ ನೀರಾವರಿ."
        },
        "harvest": {
            "en": "Harvest when leaves turn yellow.",
            "kn": "ಎಲೆಗಳು ಹಳದಿಯಾಗಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 23. CARROT
    # =====================================================
    "carrot": {
        "vegetative": {
            "en": "Maintain fine tilth; light irrigation.",
            "kn": "ಸೂಕ್ಷ್ಮ ಮಣ್ಣಿನ ಬೇಳೆ ಇರಲಿ; ಹಗುರ ನೀರಾವರಿ."
        },
        "root enlargement": {
            "en": "Ensure moisture; apply potash.",
            "kn": "ತೇವ ಇರಲಿ; ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "maturity": {
            "en": "Stop irrigation before harvest.",
            "kn": "ಕೊಯ್ತಿಗೆ ಮೊದಲು ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ."
        },
        "harvest": {
            "en": "Harvest when roots reach full size.",
            "kn": "ಮೂಲಗಳು ಪೂರ್ಣ ಗಾತ್ರ ತಲುಪಿದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 24. CAPSICUM
    # =====================================================
    "capsicum": {
        "vegetative": {
            "en": "Provide staking; apply nitrogen.",
            "kn": "ಸ್ಟೇಕಿಂಗ್ ನೀಡಿ; ನೈಟ್ರೋಜನ್ ನೀಡಿ."
        },
        "flowering": {
            "en": "Micronutrient spray; avoid moisture stress.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ; ತೇವ ಒತ್ತಡ ತಪ್ಪಿಸಿ."
        },
        "fruiting": {
            "en": "Apply potash; control thrips.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ತ್ರಿಪ್ಸ್ ನಿಯಂತ್ರಿಸಿ."
        },
        "harvest": {
            "en": "Harvest firm glossy fruits.",
            "kn": "ಗಟ್ಟಿ ಹೊಳೆಯುವ ಹಣ್ಣುಗಳನ್ನು ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 25. TURMERIC
    # =====================================================
    "turmeric": {
        "sprouting": {
            "en": "Maintain moisture; apply FYM.",
            "kn": "ತೇವ ಇರಲಿ; FYM ನೀಡಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; regular weeding.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ತೆಗೆದುಹಾಕಿ."
        },
        "rhizome development": {
            "en": "Apply potash; ensure irrigation.",
            "kn": "ಪೊಟಾಶ್ ನೀಡಿ; ನೀರಾವರಿ ಮಾಡಿ."
        },
        "maturation": {
            "en": "Reduce irrigation; leaves turn yellow.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ಎಲೆಗಳು ಹಳದಿ ಆಗುತ್ತವೆ."
        },
        "harvest": {
            "en": "Harvest 8–9 months after planting.",
            "kn": "ನೆಡುವ 8–9 ತಿಂಗಳ ನಂತರ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 26. GINGER
    # =====================================================
    "ginger": {
        "sprouting": {
            "en": "Provide shade; maintain moisture.",
            "kn": "ನೆರಳು ನೀಡಿ; ತೇವ ಇರಲಿ."
        },
        "vegetative": {
            "en": "Apply nitrogen; mulch field.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಲ್ಚಿಂಗ್ ಮಾಡಿ."
        },
        "rhizome development": {
            "en": "Apply FYM + potash.",
            "kn": "FYM + ಪೊಟಾಶ್ ನೀಡಿ."
        },
        "maturation": {
            "en": "Reduce irrigation; avoid waterlogging.",
            "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ; ನೀರಿನ ನಿಲ್ಲಿಕೆ ತಪ್ಪಿಸಿ."
        },
        "harvest": {
            "en": "Harvest 7–8 months after sowing.",
            "kn": "ಬಿತ್ತನೆ ನಂತರ 7–8 ತಿಂಗಳಲ್ಲಿ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 27. COFFEE
    # =====================================================
    "coffee": {
        "vegetative": {
            "en": "Shade regulation; apply manures.",
            "kn": "ನೆರಳು ನಿಯಂತ್ರಿಸಿ; ಗೊಬ್ಬರ ನೀಡಿ."
        },
        "flowering": {
            "en": "Provide blossom irrigation.",
            "kn": "ಬ್ಲಾಸಂ ನೀರಾವರಿ ಮಾಡಿ."
        },
        "fruiting": {
            "en": "Apply nutrients; control berry borer.",
            "kn": "ಪೋಷಕಾಂಶ ನೀಡಿ; ಬೆರಿ ಬೋರರ್ ನಿಯಂತ್ರಿಸಿ."
        },
        "harvest": {
            "en": "Harvest ripe red cherries.",
            "kn": "ಕೆಂಪು ಚೆರಿ ಹಣ್ಣುಗಳನ್ನು ಕೊಯ್ಯಿರಿ."
        }
    },

    # =====================================================
    # 28. TEA
    # =====================================================
    "tea": {
        "pruning": {
            "en": "Prune to maintain bush shape.",
            "kn": "ಬುಷ್ ಆಕಾರಕ್ಕೆ ಕತ್ತರಿಸಿ."
        },
        "flush growth": {
            "en": "Apply nitrogen; light irrigation.",
            "kn": "ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಹಗುರ ನೀರಾವರಿ."
        },
        "plucking": {
            "en": "Pluck 2 leaves and a bud.",
            "kn": "2 ಎಲೆ + ಒಂದು ಕೊಂಬೆ ಪ್ಲಕ್ ಮಾಡಿ."
        },
        "harvest": {
            "en": "Regular plucking improves yield.",
            "kn": "ನಿಯಮಿತ ಪ್ಲಕಿಂಗ್ ಉತ್ಪಾದನೆ ಹೆಚ್ಚಿಸುತ್ತದೆ."
        }
    },

    # =====================================================
    # 29. PEPPER
    # =====================================================
    "pepper": {
        "vegetative": {
            "en": "Provide support; apply FYM.",
            "kn": "ಆಧಾರ ನೀಡಿ; FYM ನೀಡಿ."
        },
        "flowering": {
            "en": "Light irrigation; provide shade.",
            "kn": "ಹಗುರ ನೀರಾವರಿ; ನೆರಳು ನೀಡಿ."
        },
        "fruit set": {
            "en": "Micronutrient spray.",
            "kn": "ಮೈಕ್ರೋನ್ಯುಟ್ರಿಯಂಟ್ಸ್ ನೀಡಿ."
        },
        "harvest": {
            "en": "Harvest when berries turn red.",
            "kn": "ಬೆರಿ ಕೆಂಪಾದಾಗ ಕೊಯಿರಿ."
        }
    },

    # =====================================================
    # 30. BETEL LEAF
    # =====================================================
    "betel": {
        "vegetative": {
            "en": "Provide shade; apply organic manure.",
            "kn": "ನೆರಳು ನೀಡಿ; ಜೈವಿಕ ಗೊಬ್ಬರ ನೀಡಿ."
        },
        "leaf development": {
            "en": "Maintain high humidity; frequent irrigation.",
            "kn": "ತೇವಾವಸ್ಥೆ ಹೆಚ್ಚಿರಲಿ; ನಿಯಮಿತ ನೀರಾವರಿ."
        },
        "harvest": {
            "en": "Pick mature leaves regularly.",
            "kn": "ಪೂರ್ಣ ಹಸಿದ ಎಲೆಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಕೊಯ್ಯಿರಿ."
        }
    }
}



def stage_recommendation_engine(crop_name: str, stage: str, lang: str) -> str:
    crop = (crop_name or "").lower()
    st = (stage or "").lower()
    if crop in STAGE_RECOMMENDATIONS and st in STAGE_RECOMMENDATIONS[crop]:
        return STAGE_RECOMMENDATIONS[crop][st][lang]
    fallback = {
        "en": f"No specific recommendation for {crop_name} at stage '{stage}'.",
        "kn": f"{crop_name} ಹಂತ '{stage}' ಗೆ ವಿಶೇಷ ಸಲಹೆ ಲಭ್ಯವಿಲ್ಲ."
    }
    return fallback[lang]


# =========================================================
# NEW MODULE: Fertilizer calculator per stage
# - Simple heuristics: N-P-K per hectare based on crop and stage.
# - If user/farm stores area in farmDetails (hectares), use that.
# - Accepts optional query like "fertilizer for 1 acre" via parsing in router (for now use farmDetails)
# =========================================================

# Baseline N-P-K (kg/ha) recommendations for stages (very simplified)
FERTILIZER_BASE = {

    # =====================================================
    # 1. PADDY
    # =====================================================
    "paddy": {
        "nursery": (20, 10, 10),
        "tillering": (60, 30, 20),
        "panicle initiation": (30, 20, 20),
        "flowering": (0, 0, 0),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 2. RAGI (Finger Millet)
    # =====================================================
    "ragi": {
        "germination": (20, 10, 10),
        "tillering": (40, 20, 20),
        "flowering": (20, 10, 20),
        "grain filling": (10, 0, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 3. MAIZE
    # =====================================================
    "maize": {
        "vegetative": (80, 40, 20),
        "tasseling": (40, 20, 20),
        "silking": (20, 10, 20),
        "grain filling": (10, 0, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 4. JOWAR
    # =====================================================
    "jowar": {
        "vegetative": (40, 20, 20),
        "booting": (20, 10, 20),
        "flowering": (10, 0, 20),
        "grain filling": (10, 0, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 5. TUR (Pigeon Pea)
    # =====================================================
    "tur": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20),
        "pod formation": (10, 10, 20),
        "maturity": (0, 0, 0)
    },

    # =====================================================
    # 6. MOONG (Green Gram)
    # =====================================================
    "moong": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20),
        "pod setting": (10, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 7. URAD
    # =====================================================
    "urad": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20),
        "pod setting": (10, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 8. GROUNDNUT
    # =====================================================
    "groundnut": {
        "vegetative": (20, 40, 40),
        "flowering": (20, 20, 20),
        "pod development": (10, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 9. SUNFLOWER
    # =====================================================
    "sunflower": {
        "vegetative": (40, 30, 20),
        "bud formation": (20, 10, 20),
        "flowering": (10, 10, 20),
        "seed filling": (10, 0, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 10. SESAME
    # =====================================================
    "sesame": {
        "vegetative": (20, 40, 20),
        "flowering": (10, 20, 20),
        "capsule setting": (10, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 11. SUGARCANE
    # =====================================================
    "sugarcane": {
        "tillering": (60, 40, 20),
        "grand growth": (80, 40, 40),
        "ripening": (20, 20, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 12. COTTON
    # =====================================================
    "cotton": {
        "vegetative": (60, 40, 20),
        "square formation": (40, 20, 20),
        "flowering": (20, 10, 20),
        "boll development": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 13. ARECANUT
    # =====================================================
    "arecanut": {
        "vegetative": (40, 40, 40),
        "flowering": (20, 20, 20),
        "nut development": (20, 20, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 14. COCONUT
    # =====================================================
    "coconut": {
        "vegetative": (40, 20, 60),
        "flowering": (20, 20, 20),
        "nut formation": (20, 20, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 15. BANANA
    # =====================================================
    "banana": {
        "vegetative": (60, 40, 40),
        "flowering": (40, 20, 40),
        "fruiting": (20, 10, 60),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 16. MANGO
    # =====================================================
    "mango": {
        "vegetative": (40, 20, 40),
        "flowering": (10, 10, 20),
        "fruit set": (20, 10, 40),
        "maturity": (0, 0, 0)
    },

    # =====================================================
    # 17. SAPOTA
    # =====================================================
    "sapota": {
        "vegetative": (40, 20, 40),
        "flowering": (20, 20, 20),
        "fruiting": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 18. GRAPES
    # =====================================================
    "grapes": {
        "pruning": (40, 20, 40),
        "flowering": (20, 20, 20),
        "fruiting": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 19. TOMATO
    # =====================================================
    "tomato": {
        "vegetative": (50, 40, 40),
        "flowering": (30, 20, 40),
        "fruiting": (20, 10, 60),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 20. BRINJAL
    # =====================================================
    "brinjal": {
        "vegetative": (40, 40, 20),
        "flowering": (20, 20, 20),
        "fruiting": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 21. ONION
    # =====================================================
    "onion": {
        "vegetative": (40, 20, 20),
        "bulb formation": (20, 20, 40),
        "maturation": (10, 0, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 22. POTATO
    # =====================================================
    "potato": {
        "vegetative": (60, 40, 40),
        "tuber initiation": (20, 20, 40),
        "bulking": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 23. CARROT
    # =====================================================
    "carrot": {
        "vegetative": (40, 20, 20),
        "root enlargement": (20, 10, 40),
        "maturity": (10, 0, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 24. CAPSICUM
    # =====================================================
    "capsicum": {
        "vegetative": (40, 40, 20),
        "flowering": (20, 20, 20),
        "fruiting": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 25. TURMERIC
    # =====================================================
    "turmeric": {
        "sprouting": (30, 20, 20),
        "vegetative": (40, 30, 30),
        "rhizome development": (20, 20, 40),
        "maturation": (10, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 26. GINGER
    # =====================================================
    "ginger": {
        "sprouting": (30, 20, 20),
        "vegetative": (40, 30, 30),
        "rhizome development": (20, 20, 40),
        "maturation": (10, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 27. COFFEE
    # =====================================================
    "coffee": {
        "vegetative": (40, 20, 40),
        "flowering": (20, 10, 20),
        "fruiting": (20, 10, 40),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 28. TEA
    # =====================================================
    "tea": {
        "pruning": (20, 20, 20),
        "flush growth": (40, 20, 20),
        "plucking": (20, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 29. PEPPER
    # =====================================================
    "pepper": {
        "vegetative": (20, 20, 20),
        "flowering": (20, 10, 20),
        "fruit set": (20, 10, 20),
        "harvest": (0, 0, 0)
    },

    # =====================================================
    # 30. BETEL LEAF
    # =====================================================
    "betel": {
        "vegetative": (20, 20, 20),
        "leaf development": (20, 10, 20),
        "harvest": (0, 0, 0)
    }
}



def fertilizer_calculator(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    # Check farm area (hectares)
    farm = get_user_farm_details(user_id)
    area_ha = None
    # farmDetails may store area as 'area' (hectares) or 'areaInHectares'; be defensive
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
# NEW MODULE: Pesticide recommendation engine
# - Map common pests/diseases to recommendations (preferred bio options included)
# =========================================================
PESTICIDE_DB = {

    # =========================================================
    # 🟢 COMMON INSECT PESTS
    # =========================================================

    "aphid": {
        "en": "Spray neem oil (2%) or insecticidal soap. Use yellow sticky traps. If severe, use approved systemic insecticide as per label.",
        "kn": "ನೀಮ್ ಎಣ್ಣೆ (2%) ಅಥವಾ ಸಾಬೂನು ಸಿಂಪಡಿಸಿ. ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್ ಬಳಸಿ. ಗಂಭೀರವಾದರೆ ಲೇಬಲ್ ಪ್ರಕಾರ ಸಿಸ್ಟಮಿಕ್ ಕೀಟನಾಶಕ ಬಳಸಿ."
    },

    "whitefly": {
        "en": "Use yellow sticky traps, neem oil (2%), introduce predators like ladybird beetles. If required, use recommended systemic insecticide.",
        "kn": "ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್, ನೀಮ್ ಎಣ್ಣೆ (2%) ಬಳಸಿ. ಲೇಡಿಬರ್ಡ್ ಕೀಟಗಳನ್ನು ಬಿಡುಗಡೆ ಮಾಡಿ. ಅವಶ್ಯಕವಾದರೆ ಶಿಫಾರಸು ಮಾಡಿದ ಸಿಸ್ಟಮಿಕ್ ಕೀಟನಾಶಕ ಬಳಸಿ."
    },

    "thrips": {
        "en": "Maintain field sanitation, spray neem oil 2%, use blue sticky traps. Apply recommended insecticide only if infestation is heavy.",
        "kn": "ಕ್ಷೇತ್ರ ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ, 2% ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ, ನೀಲಿ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್ ಬಳಸಿ. ಗಂಭೀರ ವಿಪತ್ತು ಇರೆ ಮಾತ್ರ ಶಿಫಾರಸು ಮಾಡಿದ ವಿಷರಹಿತ ಕೀಟನಾಶಕ ಬಳಸಿ."
    },

    "mites": {
        "en": "Increase humidity, apply neem oil 2%, use sulfur-based bio-miticides.",
        "kn": "ತೇವಾಂಶ ಹೆಚ್ಚಿಸಿ, 2% ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ, ಸಲ್ಪರ್ ಆಧಾರಿತ ಜೈವ ಮಿಟಿಸೈಡ್ ಬಳಸಿ."
    },

    "jassid": {
        "en": "Spray neem oil (1.5%), use sticky traps, remove weeds around field.",
        "kn": "ನೀಮ್ ಎಣ್ಣೆ (1.5%) ಸಿಂಪಡಿಸಿ, ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್ ಬಳಸಿರಿ, ಹೊಲದ ಸುತ್ತಲಿನ ಕಳೆ ತೆಗೆದುಹಾಕಿ."
    },

    "stem borer": {
        "en": "Install pheromone traps. Release Trichogramma cards. Destroy deadhearts. Apply recommended insecticide only when needed.",
        "kn": "ಫೆರೋಮೋನ್ ಟ್ರಾಪ್, ಟ್ರೈಕೋಗ್ರಾಮಾ ಕಾರ್ಡ್ ಬಳಸಿ. ಡೆಡ್‌ಹಾರ್ಟ್ ತೆಗೆದುಹಾಕಿ. ಅಗತ್ಯವಿದ್ದರೆ ಮಾತ್ರ ಕೀಟನಾಶಕ ಬಳಸಿ."
    },

    "fruit borer": {
        "en": "Use pheromone traps, install light traps. Apply Bacillus thuringiensis (Bt).",
        "kn": "ಫೆರೋಮೋನ್ ಟ್ರಾಪ್, ಲೈಟ್ ಟ್ರಾಪ್ ಬಳಸಿ. ಬ್ಯಾಸಿಲಸ್ ಥುರಿಂಜಿಯೆನ್ಸಿಸ್ (Bt) ಸಿಂಪಡಿಸಿ."
    },

    "shoot borer": {
        "en": "Remove infested shoots, use pheromone traps, and apply neem oil.",
        "kn": "ಸೋಂಕಿತ ತೊಡೆಯನ್ನು ತೆಗೆದುಹಾಕಿ, ಫೆರೋಮೋನ್ ಟ್ರಾಪ್ ಬಳಸಿ, ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ."
    },

    "armyworm": {
        "en": "Spray neem oil 2%, release Trichogramma, maintain field hygiene.",
        "kn": "2% ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ, ಟ್ರೈಕೋಗ್ರಾಮಾ ಬಿಡುಗಡೆ ಮಾಡಿ, ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ."
    },

    "hairy caterpillar": {
        "en": "Hand pick early larvae, use flame torch at night, apply neem spray.",
        "kn": "ಪ್ರಾಥಮಿಕ ಲಾರ್ವಾ ತೆಗೆದುಹಾಕಿ, ರಾತ್ರಿ ಫ್ಲೇಮ್ ಟಾರ್ಚ್ ಬಳಸಿ, ನೀಮ್ ಸಿಂಪಡಿಸಿ."
    },

    "mealybug": {
        "en": "Use soap solution, neem oil, prune infested parts, release predators (Cryptolaemus).",
        "kn": "ಸಾಬೂನು ದ್ರಾವಣ, ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ, ಸೋಂಕಿತ ಕೊಂಬೆ ಕಡಿತ ಮಾಡಿ."
    },

    # =========================================================
    # 🟠 COMMON FUNGAL DISEASES
    # =========================================================

    "blast": {
        "en": "Improve drainage, avoid excess nitrogen, apply recommended fungicide such as tricyclazole where permitted.",
        "kn": "ನೀರಿನ ನಿಃಸ್ರಾವ ಸುಧಾರಿಸಿ, ಯೂರಿಯಾ ಅತಿ ಬಳಕೆ ತಪ್ಪಿಸಿ, ಶಿಫಾರಸು ಮಾಡಿದ ಫಂಗಿಸೈಡ್ ಬಳಸಿ."
    },

    "powdery mildew": {
        "en": "Use sulfur dusting, spray neem oil, apply potassium bicarbonate.",
        "kn": "ಸಲ್ಪರ್ ಧೂಳು ಹಾಕಿ, ನೀಮ್ ಎಣ್ಣೆ ಬಳಸಿ, ಪೊಟಾಶಿಯಂ ಬೈಕಾರ್ಬೊನೆಟ್ ಸಿಂಪಡಿಸಿ."
    },

    "downy mildew": {
        "en": "Ensure airflow, avoid overhead irrigation, apply copper-based fungicides.",
        "kn": "ಗಾಳಿ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ, ಮೇಲಿನಿಂದ ನೀರಾವರಿ ತಪ್ಪಿಸಿ, ಕಾಪರ್ ಆಧಾರಿತ ಔಷಧ ಬಳಸಿ."
    },

    "wilt": {
        "en": "Use Trichoderma in soil, improve drainage, avoid waterlogging.",
        "kn": "ಟ್ರೈಕೊಡರ್ಮಾ ಮಣ್ಣಿಗೆ ನೀಡಿ, ನೀರಿನ ನಿಃಸ್ರಾವ ಸುಧಾರಿಸಿ."
    },

    "root rot": {
        "en": "Improve drainage, use Trichoderma, avoid excess moisture.",
        "kn": "ನೀರಿನ ನಿಃಸ್ರಾವ ಉತ್ತಮಗೊಳಿಸಿ, ಟ್ರೈಕೊಡರ್ಮಾ ಬಳಸಿ."
    },

    "leaf spot": {
        "en": "Remove infected leaves, improve ventilation, spray neem or copper oxychloride.",
        "kn": "ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಗಾಳಿ ಸಂಚಾರ ಸುಧಾರಿಸಿ."
    },

    "anthracnose": {
        "en": "Apply neem extract, prune infected twigs, use biofungicide.",
        "kn": "ನೀಮ್ ಎಕ್ಸ್‌ಟ್ರಾಕ್ಟ್ ಬಳಸಿ, ಸೋಂಕಿತ ಕೊಂಬೆ ತೆಗೆದುಹಾಕಿ."
    },

    # =========================================================
    # 🔵 COMMON BACTERIAL DISEASES
    # =========================================================

    "bacterial blight": {
        "en": "Use disease-free seeds, avoid overhead irrigation, apply approved copper bactericides.",
        "kn": "ರೋಗ ರಹಿತ ಬೀಜ ಬಳಸಿ, ಮೇಲಿನ ನೀರಾವರಿ ತಪ್ಪಿಸಿ, ಕಾಪರ್ ಬ್ಯಾಕ್ಟೆರಿಸೈಡ್ ಬಳಸಿ."
    },

    "leaf blight": {
        "en": "Remove infected leaves, maintain spacing, apply copper fungicide.",
        "kn": "ಸೋಂಕಿತ ಎಲೆ ತೆಗೆದುಹಾಕಿ, ಸರಿಯಾದ ಅಂತರ ಕಾಪಾಡಿ."
    },

    "soft rot": {
        "en": "Improve drainage, avoid injury, apply bleaching powder around base.",
        "kn": "ನೀರಿನ ನಿಃಸ್ರಾವ ಉತ್ತಮವಾಗಿರಲಿ, ಸಸ್ಯಕ್ಕೆ ಗಾಯ ತಪ್ಪಿಸಿ."
    },

    # =========================================================
    # 🔴 VIRAL DISEASES
    # =========================================================

    "leaf curl": {
        "en": "Caused by whiteflies. Control whiteflies, remove infected plants, use neem oil.",
        "kn": "ವೈಟ್‌ಫ್ಲೈ ಕಾರಣ. ವೈಟ್‌ಫ್ಲೈ ನಿಯಂತ್ರಿಸಿ, ಸೋಂಕಿತ ಸಸ್ಯ ತೆಗೆದುಹಾಕಿ."
    },

    "mosaic virus": {
        "en": "Remove infected plants, control aphids/whiteflies, grow resistant varieties.",
        "kn": "ಸೋಂಕಿತ ಸಸ್ಯ ತೆಗೆದುಹಾಕಿ, ಆಫಿಡ್/ವೈಟ್‌ಫ್ಲೈ ನಿಯಂತ್ರಿಸಿ."
    },

    "bud necrosis": {
        "en": "Thrips control is key. Remove infected plants, spray neem oil.",
        "kn": "ಥ್ರಿಪ್ಸ್ ನಿಯಂತ್ರಣ ಮುಖ್ಯ. ಸೋಂಕಿತ ಸಸ್ಯ ತೆಗೆದುಹಾಕಿ."
    },

    # =========================================================
    # 🟣 NEMATODE ISSUES
    # =========================================================

    "root knot nematode": {
        "en": "Apply neem cake, use bio-nematicides (Paecilomyces, Purpureocillium), rotate crops.",
        "kn": "ನೀಮ್ ಕೆಕ್ ನೀಡಿ, ಜೈವ ನೆಮಾಟಿಸೈಡ್ ಬಳಸಿ, ಬೆಳೆ ಬದಲಾವಣೆ ಮಾಡಿ."
    },

    # =========================================================
    # ⭐ SPECIAL CROP-SPECIFIC ISSUES
    # =========================================================

    "sigatoka": {  # Banana
        "en": "Improve aeration, remove infected leaves, apply recommended fungicide.",
        "kn": "ಗಾಳಿ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ, ಸೋಂಕಿತ ಎಲೆ ತೆಗೆದುಹಾಕಿ."
    },

    "tungro": { # Paddy viral disease
        "en": "Control green leafhopper. Remove infected clumps. Use resistant varieties.",
        "kn": "ಗ್ರೀನ್ ಲೀಫ್ಹಾಪರ್ ನಿಯಂತ್ರಿಸಿ. ಸೋಂಕಿತ ಸಸ್ಯ ತೆಗೆದುಹಾಕಿ."
    },

    "red palm weevil": { # Coconut & arecanut
        "en": "Use pheromone traps, avoid injuries to trunk, remove infested trees early.",
        "kn": "ಫೆರೋಮೋನ್ ಟ್ರಾಪ್ ಬಳಸಿ, ಕಡ್ಡಿಗೆ ಗಾಯ ತಪ್ಪಿಸಿ."
    },

    "berry borer": { # Coffee
        "en": "Hand pick infested berries, strip harvest, use pheromone traps.",
        "kn": "ಸೋಂಕಿತ ಕಾಯಿ ತೆಗೆದುಹಾಕಿ, ಫೆರೋಮೋನ್ ಟ್ರಾಪ್ ಬಳಸಿ."
    },

    "dieback": { # Mango
        "en": "Prune diseased branches, apply copper fungicide, improve aeration.",
        "kn": "ಸೋಂಕಿತ ಕೊಂಬೆ ಕಡಿತ ಮಾಡಿ, ಕಾಪರ್ ಔಷಧ ಬಳಸಿ."
    },

    "pink bollworm": { # Cotton
        "en": "Use pheromone traps, remove rosette flowers, avoid late sowing.",
        "kn": "ಫೆರೋಮೋನ್ ಟ್ರಾಪ್ ಬಳಸಿ, ಅಪಾಯದ ಹೂ ತೆಗೆದುಹಾಕಿ."
    },

    "rust": {
        "en": "Use sulfur dusting, neem spray, improve spacing.",
        "kn": "ಸಲ್ಪರ್ ಧೂಳು, ನೀಮ್ ಸಿಂಪಡಣೆ, ಅಂತರ ಕಾಪಾಡಿ."
    }
}



def pesticide_recommendation(crop: str, pest: str, lang: str) -> Tuple[str, bool, List[str]]:
    pest_l = (pest or "").lower()
    if pest_l in PESTICIDE_DB:
        return PESTICIDE_DB[pest_l][lang if lang in ["en", "kn"] else "en"], False, ["Use bio-pesticide", "Contact advisor"]
    # fuzzy match: check substring
    for key in PESTICIDE_DB.keys():
        if key in pest_l:
            return PESTICIDE_DB[key][lang if lang in ["en", "kn"] else "en"], False, ["Use bio-pesticide", "Contact advisor"]
    fallback = {
        "en": "Pest not recognized. Provide photo or pest name (e.g., 'aphid', 'fruit borer').",
        "kn": "ಕೀಟ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ಫೋಟೋ ಅಥವಾ ಕೀಟದ ಹೆಸರು ನೀಡಿ (ಉದಾ: aphid)."
    }
    return fallback[lang], False, ["Upload photo", "Contact Krishi Adhikari"]


# =========================================================
# NEW MODULE: Irrigation schedule module
# - Suggest irrigation frequency/amount based on crop, stage, soil type, and simple weather forecast (mock)
# =========================================================
SOIL_WATER_HOLDING = {
    "sandy": 0.6,  # relative quick dry -> irrigate more
    "loamy": 1.0,
    "clay": 1.2
}

CROP_ET_BASE = {

    # --- Cereals & Millets ---
    "paddy": 6.0,         # flooded crop, high ET
    "ragi": 4.5,
    "maize": 5.5,
    "jowar": 4.8,

    # --- Pulses ---
    "tur": 4.0,
    "moong": 3.8,
    "urad": 3.8,

    # --- Oilseeds ---
    "groundnut": 4.2,
    "sunflower": 5.0,
    "sesame": 4.0,

    # --- Commercial Crops ---
    "sugarcane": 7.0,     # highest ET of field crops
    "cotton": 5.0,

    # --- Plantations ---
    "arecanut": 5.5,
    "coconut": 6.0,
    "coffee": 4.5,
    "tea": 4.0,
    "pepper": 4.0,
    "betel": 3.5,

    # --- Fruits ---
    "banana": 6.5,
    "mango": 4.0,
    "sapota": 4.5,
    "grapes": 4.2,

    # --- Vegetables ---
    "tomato": 4.8,
    "brinjal": 4.5,
    "onion": 4.0,
    "potato": 4.5,
    "carrot": 4.2,
    "capsicum": 4.5,

    # --- Spices ---
    "turmeric": 5.0,
    "ginger": 5.0
}



def fetch_weather_by_location(district: str):
    """Fetch current weather from OpenWeather API."""
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"q={district}&appid={OPENWEATHER_KEY}&units=metric"
        )
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
    except:
        return None

def weather_suggestion_engine(weather, crop_stage=None, language="en"):
    temp = weather["temp"]
    humidity = weather["humidity"]
    wind = weather["wind"]
    rain = weather["rain"]
    cond = weather["condition"]

    suggestions = []

    # Temperature Logic
    if temp > 35:
        suggestions.append("High heat – give afternoon irrigation and mulch.")
    elif temp < 15:
        suggestions.append("Low temperature – avoid fertilizer today.")

    # Rain Logic
    if rain > 3:
        suggestions.append("Rainfall occurring – stop irrigation for 24 hours.")
    else:
        suggestions.append("No rain – irrigation recommended today.")

    # Humidity Logic
    if humidity > 80:
        suggestions.append("High humidity – fungal disease chances are high.")
    elif humidity < 35:
        suggestions.append("Low humidity – increase irrigation frequency.")

    # Wind Logic
    if wind > 20:
        suggestions.append("High wind – avoid spraying pesticides.")

    # Crop-stage weather fusion
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
        msg = {
            "en": "Farm district missing. Update farm details.",
            "kn": "ಫಾರಂ ಜಿಲ್ಲೆಯ ಮಾಹಿತಿ ಇಲ್ಲ. farmDetails ನವೀಕರಿಸಿ."
        }
        return msg[language], [], False

    district = farm["district"]
    weather = fetch_weather_by_location(district)

    if not weather:
        return ("Unable to fetch weather data.", [], False)

    suggestions = weather_suggestion_engine(weather, None, language)

    if language == "kn":
        report = (
            f"{district} ಹವಾಮಾನ:\n"
            f"ಸ್ಥಿತಿ: {weather['description']}\n"
            f"ತಾಪಮಾನ: {weather['temp']}°C\n"
            f"ತೇವಾಂಶ: {weather['humidity']}%\n"
            f"ಗಾಳಿ: {weather['wind']} km/h\n"
            f"ಮಳೆ (1h): {weather['rain']} mm\n"
        )
    else:
        report = (
            f"Weather in {district}:\n"
            f"Condition: {weather['description']}\n"
            f"Temperature: {weather['temp']}°C\n"
            f"Humidity: {weather['humidity']}%\n"
            f"Wind: {weather['wind']} km/h\n"
            f"Rain (1h): {weather['rain']} mm\n"
        )

    return report, suggestions, True


def irrigation_schedule(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    soil = (farm.get("soilType") or "loamy").lower()
    area_ha = None
    if isinstance(farm, dict):
        try:
            area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0)
        except Exception:
            area_ha = 1.0
    else:
        area_ha = 1.0

    district = farm.get("district") or "unknown"
    weather = get_mock_weather_for_district(district)
    rain_next_24 = weather.get("rain_next_24h_mm", 0)

    crop_l = (crop or "").lower()
    base_et = CROP_ET_BASE.get(crop_l, 4)  # mm/day default
    soil_factor = SOIL_WATER_HOLDING.get(soil, 1.0)

    # Stage-based multiplier
    stage_mult = 1.0
    if "nursery" in (stage or "").lower() or "vegetative" in (stage or "").lower():
        stage_mult = 1.2
    elif "flower" in (stage or "").lower() or "panicle" in (stage or "").lower():
        stage_mult = 1.1
    elif "harvest" in (stage or "").lower():
        stage_mult = 0.8

    # Compute required irrigation mm/day (very simplified)
    required_mm = base_et * stage_mult * (1.0 / soil_factor)
    if rain_next_24 >= 10:
        suggestion = {
            "en": "Rain expected soon. Delay irrigation and monitor soil moisture.",
            "kn": "ಶೀಘ್ರದಲ್ಲೇ ಮಳೆಯ ಸಂಭವನೆ. ನೀರಾವರಿ ತಡೆಯಿರಿ ಮತ್ತು ಮಣ್ಣು ஈರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿರಿ."
        }
        return suggestion[lang], False, ["Soil moisture check", "Delay irrigation"]

    # convert mm/day to liters per ha: 1 mm = 10,000 liters per hectare
    liters_per_ha = required_mm * 10000
    total_liters = round(liters_per_ha * area_ha, 1)
    if lang == "kn":
        text = (f"{crop.title()} ({stage}) - ಶಿಫಾರಸು: ಪ್ರತಿ ದಿನ {round(required_mm,1)} mm ನೀರಾವರಿ (ಸಮಾನವಾದ ~{total_liters} ಲೀಟರ್/ದಿನಕ್ಕೆ {area_ha} ha).")
    else:
        text = (f"Recommendation for {crop.title()} ({stage}): approx {round(required_mm,1)} mm/day irrigation "
                f"(~{total_liters} liters/day for {area_ha} ha). Adjust if rain or soil moisture indicates otherwise.")
    return text, False, ["Soil moisture sensor", "Irrigation logs"]


# =========================================================
# NEW MODULE: Simple Yield prediction (heuristic)
# - Uses base yield per crop and multipliers from fertilizer/irrigation/pest control flags.
# - In production use statistical model with historical data.
# =========================================================

BASE_YIELD_TON_PER_HA = {

    # --- Cereals & Millets ---
    "paddy": 4.0,            # irrigated condition
    "ragi": 1.8,
    "maize": 3.5,
    "jowar": 1.2,

    # --- Pulses ---
    "tur": 1.0,
    "moong": 0.8,
    "urad": 0.8,

    # --- Oilseeds ---
    "groundnut": 1.5,
    "sunflower": 1.0,
    "sesame": 0.6,

    # --- Commercial Crops ---
    "sugarcane": 80.0,        # t/ha (sugarcane measured as cane yield)
    "cotton": 1.2,            # lint yield

    # --- Plantations ---
    "arecanut": 2.0,
    "coconut": 10.0,          # nuts converted to t/ha equivalent
    "coffee": 0.8,
    "tea": 2.0,
    "pepper": 1.0,
    "betel": 4.0,

    # --- Fruits ---
    "banana": 20.0,
    "mango": 8.0,
    "sapota": 15.0,
    "grapes": 12.0,

    # --- Vegetables ---
    "tomato": 25.0,
    "brinjal": 20.0,
    "onion": 18.0,
    "potato": 22.0,
    "carrot": 30.0,
    "capsicum": 18.0,

    # --- Spices ---
    "turmeric": 8.0,
    "ginger": 15.0
}



def yield_prediction(crop: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    """
    Uses simple heuristics: base yield * factors (fertilizer adequacy, irrigation adequacy, pest control)
    Tries to read simple indicators from Firebase: lastFertilizerApplied, irrigationLogs, pestIncidents
    """
    farm = get_user_farm_details(user_id)
    area_ha = None
    if isinstance(farm, dict):
        try:
            area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0)
        except Exception:
            area_ha = 1.0
    else:
        area_ha = 1.0

    crop_l = (crop or "").lower()
    base = BASE_YIELD_TON_PER_HA.get(crop_l, 2.0)

    # read basic indicators
    last_fert = firebase_get(f"Users/{user_id}/lastFertilizerApplied") or {}
    fert_ok = False
    if isinstance(last_fert, dict) and last_fert.get("applied", False):
        fert_ok = True

    irrigation_logs = firebase_get(f"Users/{user_id}/irrigationLogs") or {}
    irrigation_ok = False
    # if irrigation logs in last 14 days exist -> ok
    if isinstance(irrigation_logs, dict):
        found_recent = False
        now = datetime.utcnow().timestamp()
        for k, v in irrigation_logs.items():
            ts = v.get("timestamp", 0)
            if now - ts < 14 * 24 * 3600:
                found_recent = True
                break
        irrigation_ok = found_recent

    pest_incidents = firebase_get(f"Users/{user_id}/pestIncidents") or {}
    pest_control_ok = True
    if isinstance(pest_incidents, dict) and len(pest_incidents) > 0:
        # if recent major incidents present, reduce yield
        pest_control_ok = False

    # factor multipliers
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
# CROP–DISEASE PREDICTION FROM WEATHER (Temp/Humidity/Rain)
# =========================================================

DISEASE_WEATHER_RISK = {

    # ============================================================
    # 1. PADDY
    # ============================================================
    "paddy": [
        {"cond": "high_humidity", "disease": "blast"},
        {"cond": "continuous_rain", "disease": "bacterial blight"},
        {"cond": "high_temp_low_humidity", "disease": "brown spot"},
        {"cond": "rainy", "disease": "sheath blight"}
    ],

    # ============================================================
    # 2. RAGI (Finger Millet)
    # ============================================================
    "ragi": [
        {"cond": "high_humidity", "disease": "blast"},
        {"cond": "rainy", "disease": "leaf spot"},
        {"cond": "high_temp", "disease": "root rot"}
    ],

    # ============================================================
    # 3. MAIZE
    # ============================================================
    "maize": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "high_temp_low_humidity", "disease": "leaf blight"},
        {"cond": "rainy", "disease": "stem rot"}
    ],

    # ============================================================
    # 4. JOWAR (Sorghum)
    # ============================================================
    "jowar": [
        {"cond": "high_humidity", "disease": "anthracnose"},
        {"cond": "high_temp", "disease": "charcoal rot"},
        {"cond": "rainy", "disease": "grain mold"}
    ],

    # ============================================================
    # 5. TUR (Pigeon Pea)
    # ============================================================
    "tur": [
        {"cond": "high_humidity", "disease": "wilt"},
        {"cond": "rainy", "disease": "pod borer infestation"},
        {"cond": "high_temp", "disease": "stem canker"}
    ],

    # ============================================================
    # 6. MOONG (Green Gram)
    # ============================================================
    "moong": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "rainy", "disease": "anthracnose"},
        {"cond": "high_temp_low_humidity", "disease": "yellow mosaic virus"}
    ],

    # ============================================================
    # 7. URAD
    # ============================================================
    "urad": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "rainy", "disease": "anthracnose"},
        {"cond": "high_temp", "disease": "YMV (Yellow Mosaic Virus)"}
    ],

    # ============================================================
    # 8. GROUNDNUT
    # ============================================================
    "groundnut": [
        {"cond": "high_humidity", "disease": "late leaf spot"},
        {"cond": "rainy", "disease": "rust"},
        {"cond": "high_temp", "disease": "root rot"}
    ],

    # ============================================================
    # 9. SUNFLOWER
    # ============================================================
    "sunflower": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "rainy", "disease": "stem rot"},
        {"cond": "high_temp", "disease": "powdery mildew"}
    ],

    # ============================================================
    # 10. SESAME
    # ============================================================
    "sesame": [
        {"cond": "rainy", "disease": "phyllody"},
        {"cond": "high_humidity", "disease": "leaf spot"},
        {"cond": "high_temp_low_humidity", "disease": "stem rot"}
    ],

    # ============================================================
    # 11. SUGARCANE
    # ============================================================
    "sugarcane": [
        {"cond": "high_humidity", "disease": "red rot"},
        {"cond": "high_temp", "disease": "pokkah boeng"},
        {"cond": "continuous_rain", "disease": "ratoon stunting disease"}
    ],

    # ============================================================
    # 12. COTTON
    # ============================================================
    "cotton": [
        {"cond": "high_temp", "disease": "leaf curl virus"},
        {"cond": "rainy", "disease": "bacterial blight"},
        {"cond": "high_humidity", "disease": "anthracnose"}
    ],

    # ============================================================
    # 13. ARECANUT
    # ============================================================
    "arecanut": [
        {"cond": "high_humidity", "disease": "koleroga (fruit rot)"},
        {"cond": "heavy_rain", "disease": "bud rot"},
        {"cond": "high_temp", "disease": "yellow leaf disease"}
    ],

    # ============================================================
    # 14. COCONUT
    # ============================================================
    "coconut": [
        {"cond": "rainy", "disease": "stem bleeding"},
        {"cond": "high_humidity", "disease": "bud rot"},
        {"cond": "high_temp", "disease": "mite infestation"}
    ],

    # ============================================================
    # 15. BANANA
    # ============================================================
    "banana": [
        {"cond": "high_humidity", "disease": "sigatoka leaf spot"},
        {"cond": "rainy", "disease": "panama wilt"},
        {"cond": "high_temp", "disease": "bunchy top virus"}
    ],

    # ============================================================
    # 16. MANGO
    # ============================================================
    "mango": [
        {"cond": "rainy", "disease": "anthracnose"},
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "high_temp", "disease": "dieback"}
    ],

    # ============================================================
    # 17. SAPOTA
    # ============================================================
    "sapota": [
        {"cond": "high_humidity", "disease": "leaf spot"},
        {"cond": "rainy", "disease": "fruit rot"},
        {"cond": "high_temp", "disease": "mite infestation"}
    ],

    # ============================================================
    # 18. GRAPES
    # ============================================================
    "grapes": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "rainy", "disease": "downy mildew"},
        {"cond": "high_temp", "disease": "sunburn & berry cracking"}
    ],

    # ============================================================
    # 19. TOMATO
    # ============================================================
    "tomato": [
        {"cond": "high_humidity", "disease": "late blight"},
        {"cond": "rainy", "disease": "early blight"},
        {"cond": "high_temp", "disease": "leaf curl virus"}
    ],

    # ============================================================
    # 20. BRINJAL
    # ============================================================
    "brinjal": [
        {"cond": "high_humidity", "disease": "phomopsis blight"},
        {"cond": "rainy", "disease": "bacterial wilt"},
        {"cond": "high_temp", "disease": "shoot & fruit borer prevalence"}
    ],

    # ============================================================
    # 21. ONION
    # ============================================================
    "onion": [
        {"cond": "high_humidity", "disease": "downy mildew"},
        {"cond": "rainy", "disease": "purple blotch"},
        {"cond": "high_temp", "disease": "basal rot"}
    ],

    # ============================================================
    # 22. POTATO
    # ============================================================
    "potato": [
        {"cond": "high_humidity", "disease": "late blight"},
        {"cond": "rainy", "disease": "early blight"},
        {"cond": "high_temp", "disease": "tuber cracking"}
    ],

    # ============================================================
    # 23. CARROT
    # ============================================================
    "carrot": [
        {"cond": "high_humidity", "disease": "leaf blight"},
        {"cond": "rainy", "disease": "root rot"},
        {"cond": "high_temp", "disease": "nematode attack"}
    ],

    # ============================================================
    # 24. CAPSICUM
    # ============================================================
    "capsicum": [
        {"cond": "high_humidity", "disease": "powdery mildew"},
        {"cond": "rainy", "disease": "bacterial spot"},
        {"cond": "high_temp", "disease": "sun scald"}
    ],

    # ============================================================
    # 25. TURMERIC
    # ============================================================
    "turmeric": [
        {"cond": "high_humidity", "disease": "leaf blotch"},
        {"cond": "rainy", "disease": "rhizome rot"},
        {"cond": "high_temp", "disease": "leaf scorch"}
    ],

    # ============================================================
    # 26. GINGER
    # ============================================================
    "ginger": [
        {"cond": "high_humidity", "disease": "soft rot"},
        {"cond": "rainy", "disease": "rhizome rot"},
        {"cond": "high_temp", "disease": "leaf spot"}
    ],

    # ============================================================
    # 27. COFFEE
    # ============================================================
    "coffee": [
        {"cond": "high_humidity", "disease": "leaf rust"},
        {"cond": "rainy", "disease": "berry disease"},
        {"cond": "high_temp", "disease": "white stem borer"}
    ],

    # ============================================================
    # 28. TEA
    # ============================================================
    "tea": [
        {"cond": "high_humidity", "disease": "blister blight"},
        {"cond": "rainy", "disease": "root rot"},
        {"cond": "high_temp", "disease": "mite attack"}
    ],

    # ============================================================
    # 29. PEPPER
    # ============================================================
    "pepper": [
        {"cond": "high_humidity", "disease": "quick wilt"},
        {"cond": "rainy", "disease": "foot rot"},
        {"cond": "high_temp", "disease": "yellowing disease"}
    ],

    # ============================================================
    # 30. BETEL LEAF
    # ============================================================
    "betel": [
        {"cond": "high_humidity", "disease": "foot rot"},
        {"cond": "rainy", "disease": "leaf spot"},
        {"cond": "high_temp", "disease": "anthracnose"}
    ]
}
def classify_weather_condition(weather):
    temp = weather["temp"]
    humidity = weather["humidity"]
    rain = weather["rain"]

    conds = []

    if humidity > 80:
        conds.append("high_humidity")

    if temp > 32 and humidity < 50:
        conds.append("high_temp_low_humidity")

    if temp > 34:
        conds.append("high_temp")

    if rain > 2:
        conds.append("rainy")

    if rain > 8:
        conds.append("continuous_rain")
        conds.append("heavy_rain")

    return conds
def predict_disease_from_weather(crop, weather, lang):
    crop = crop.lower()
    if crop not in DISEASE_WEATHER_RISK:
        return None  # No model available

    weather_conditions = classify_weather_condition(weather)
    risks = []

    for rule in DISEASE_WEATHER_RISK[crop]:
        if rule["cond"] in weather_conditions:
            risks.append(rule["disease"])

    if not risks:
        msg = {
            "en": f"No major disease risk predicted for {crop.title()} based on current weather.",
            "kn": f"ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಆಧಾರದಲ್ಲಿ {crop} ಬೆಳೆಗೂ ಮುಖ್ಯ ರೋಗ ಅಪಾಯ ಕಂಡುಬಂದಿಲ್ಲ."
        }
        return msg[lang]

    # Build response
    if lang == "kn":
        text = f"{crop} ಬೆಳೆ ಹವಾಮಾನ ಆಧಾರಿತ ರೋಗ ಅಪಾಯ:\n\n"
        for d in risks:
            text += f"⚠ {d} ಅಪಾಯ ಹೆಚ್ಚು\n"
        text += "\nತಡೆ ಕ್ರಮ: ನೀಮ್ ಸಿಂಪಡಣೆ / ಗಾಳಿ ಸಂಚಾರ ಹೆಚ್ಚಿಸಿ / ಜಲಾವೃತ ತಪ್ಪಿಸಿ."
    else:
        text = f"Disease Risk Prediction for {crop.title()}:\n\n"
        for d in risks:
            text += f"⚠ High risk of {d}\n"
        text += "\nPreventive actions: Neem spray / Improve aeration / Avoid waterlogging."

    return text
# =========================================================
#NEW MODULE :SYMPTOM RECOGNITION
# =========================================================
# Symptom canonicalization -> list of canonical symptom keys
SYMPTOM_DB = {
    # single-word or short-phrase canonical symptoms
    "yellow leaves": ["nutrient deficiency", "nitrogen deficiency", "leaf curl virus", "wilt"],
    "leaf curling": ["leaf curl virus", "thrips", "aphid", "whitefly"],
    "white powder": ["powdery mildew"],
    "black spots": ["leaf spot", "early blight", "anthracnose"],
    "holes in leaves": ["caterpillar", "armyworm", "grasshopper"],
    "small holes in fruits": ["fruit borer", "borer"],
    "sticky honeydew": ["aphid", "whitefly", "mealybug"],
    "wilting": ["vascular wilt", "root rot", "phytophthora"],
    "root rot": ["root rot", "phytophthora", "rhizoctonia"],
    "brown spots": ["brown spot", "leaf spot", "blast"],
    "webbing": ["mite"],
    "tiny insects": ["aphid", "whitefly", "thrips"],
    "whiteflies": ["whitefly"],
    "aphids": ["aphid"],
    "thrips": ["thrips"],
    "spots on leaves": ["leaf spot", "early blight"],
    "fruit rot": ["anthracnose", "sigatoka", "fruit rot"],
    "leaf blight": ["blight", "bacterial blight", "early blight"],
    "soft rot": ["soft rot", "bacterial soft rot"],
    "powdery": ["powdery mildew"],
    "yellowing and spots": ["virus", "leaf spot", "nutrient deficiency"],
    "brown patches": ["leaf spot", "nutrient burn"],
    "stem bore": ["stem borer", "borer"],
    "chewed leaves": ["caterpillar", "grasshopper"],
    "white webbing": ["mite"],
    "small black dots": ["thrips", "mite", "spot"],
    "holes in fruits": ["fruit borer"],
    "leaf rolling": ["leaf curl virus", "jassid", "thrips"]
}
SYMPTOM_SYNONYMS = {
    "yellowing": "yellow leaves",
    "yellow leaves": "yellow leaves",
    "leaf curl": "leaf curling",
    "leaves curled": "leaf curling",
    "curling leaves": "leaf curling",
    "white powder on leaves": "white powder",
    "white powdery": "white powder",
    "black spots on leaf": "black spots",
    "holes in leaf": "holes in leaves",
    "holes in fruit": "holes in fruits",
    "honeydew": "sticky honeydew",
    "sticky stuff": "sticky honeydew",
    "webbing on leaves": "webbing",
    "tiny bugs": "tiny insects",
    "brown spots": "brown spots",
    "leaf spots": "spots on leaves",
    "fruit rot": "fruit rot",
    "soft rot": "soft rot",
    "stem borer": "stem bore",
    "yellow and curling": "yellowing and spots",
    # extend with more common farmer phrases as needed
}
CROP_SYMPTOM_WEIGHT = {
    "paddy": {"tungro": 2.0, "blast": 1.8, "brown spot": 1.5, "leaf blight": 1.4, "stem borer": 1.6, "leaf curl virus": 1.0},
    "tomato": {"late blight": 2.0, "early blight": 1.8, "anthracnose": 1.6, "leaf spot": 1.4, "fruit borer": 1.3},
    "chilli": {"fruit borer": 1.9, "anthracnose": 1.7, "leaf curl virus": 1.8},
    "cotton": {"pink bollworm": 2.0, "leaf curl": 1.6},
    "banana": {"sigatoka": 2.0, "panama wilt": 1.6},
    "arecanut": {"bud rot": 2.0, "koleroga (fruit rot)": 1.9}
    # extend as needed for other crops
}
DISEASE_META = {
    "leaf curl virus": {"type": "viral", "note": "Usually transmitted by whiteflies"},
    "aphid": {"type": "insect", "note": "Sucking insect - causes honeydew"},
    "whitefly": {"type": "insect", "note": "Sucking insect - transmits viruses"},
    "powdery mildew": {"type": "fungal", "note": "White powder on leaf surfaces"},
    "leaf spot": {"type": "fungal", "note": "Dark spots on leaves"},
    "fruit borer": {"type": "insect", "note": "Holes in fruits, bored fruit interior"},
    "stem borer": {"type": "insect", "note": "Internal stem damage, dead hearts"},
    "root rot": {"type": "fungal", "note": "Roots rotten after waterlogging"},
    "anthracnose": {"type": "fungal", "note": "Fruit rot, sunken lesions"}
    # add more meta as needed
}
def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str):
    return text.split()


# match symptom phrases (exact, synonym, fuzzy)
def _extract_symptom_keys(user_text: str, fuzzy_threshold: float = 0.6):
    """
    Returns list of matched canonical symptom keys (may contain duplicates).
    Uses:
      - exact substring match against synonyms and canonical keys
      - fuzzy matching via difflib for partial matches
    """
    text = _normalize_text(user_text)
    found = []

    # check synonyms first (longest match priority)
    for phrase, canonical in sorted(SYMPTOM_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in text:
            found.append(canonical)

    # check canonical keys exact substring
    for key in SYMPTOM_DB.keys():
        if key in text:
            found.append(key)

    # fuzzy match short phrases -> to capture variants
    tokens = _tokenize(text)
    joined = " ".join(tokens)
    for key in SYMPTOM_DB.keys():
        ratio = difflib.SequenceMatcher(None, joined, key).ratio()
        if ratio >= fuzzy_threshold:
            found.append(key)

    # additional n-gram matching: 2-4 gram window
    n = len(tokens)
    for L in range(2, min(6, n+1)):
        for i in range(n - L + 1):
            gram = " ".join(tokens[i:i+L])
            for phrase, canonical in SYMPTOM_SYNONYMS.items():
                if gram == phrase:
                    found.append(canonical)

    return list(found)

# scoring engine
def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
    """
    For each matched symptom key, map to candidate diseases/pests from SYMPTOM_DB,
    and accumulate weighted scores. Crop-specific boosts applied if crop known.
    Returns list of (candidate, score, evidence_list).
    """
    scores = defaultdict(float)
    evidence = defaultdict(list)

    for sk in symptom_keys:
        mapped = SYMPTOM_DB.get(sk, [])
        for cand in mapped:
            # base weight: more specific symptom -> higher base
            base_weight = 1.0
            # boost for longer/more specific keys
            if len(sk.split()) >= 2:
                base_weight += 0.25
            scores[cand] += base_weight
            evidence[cand].append(f"symptom:{sk}")

    # crop boost
    if crop:
        crop_l = crop.lower()
        crop_map = CROP_SYMPTOM_WEIGHT.get(crop_l, {})
        for cand, boost in crop_map.items():
            # only apply if candidate exists in scoring or known disease
            scores[cand] += boost
            evidence[cand].append(f"crop_boost:{crop_l}")

    # normalize & turn into sorted list
    if not scores:
        return []

    # convert to sorted list of tuples (candidate, score, evidence)
    total = sum(scores.values())
    ranked = []
    for cand, sc in sorted(scores.items(), key=lambda x: -x[1]):
        confidence = round(min(0.99, sc / (total + 1e-6)), 2)  # simplistic confidence
        ranked.append((cand, round(sc, 2), confidence, evidence.get(cand, [])))

    return ranked



def diagnose_pest(user_text, language):
    matches = match_symptoms(user_text)
    if not matches:
        fallback = {
            "en": "I could not identify the pest from the symptoms. Please describe more clearly or send a photo.",
            "kn": "ಲಕ್ಷಣಗಳಿಂದ ಕೀಟವನ್ನು ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಕಳುಹಿಸಿ."
        }
        return fallback[language], ["Upload photo", "Show common pest symptoms"]

    # Take top 1–2 matches
    result = matches[:2]

    response = f"Possible issues based on symptoms:\n" + "\n".join(f"- {d}" for d in result)
    suggestions = ["Pesticide recommendations", "Prevention steps", "Check crop stage"]

    return response, suggestions
# final diagnose function (public)
def diagnose_advanced(user_text: str, user_crop: Optional[str] = None, lang: str = "en") -> Tuple[str, bool, list]:
    """
    Input:
      - user_text: free text from farmer describing symptoms
      - user_crop: optional crop name (from farm logs). If present, used for weighting
      - lang: "en" or "kn" (language for fallback messages)
    Output:
      - response_text: a diagnostic message with top candidates, confidence & rationale
      - voice_flag: whether this response could be read by TTS (we return False default)
      - suggestions: list of actions (e.g., "Upload photo", "Pesticide recommendations", "Soil test")
    """
    if not user_text or not user_text.strip():
        fallback = {"en": "Please describe the symptoms (leaf color, spots, pests seen, part affected).",
                    "kn": "ದಯವಿಟ್ಟು ಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ (ಎಲೆ ಬಣ್ಣ, ಕಲೆ, ಕಂಡ ಹಾಳುಕಾರಕಗಳು, ಭಾಗ ಪ್ರಭಾವಿತವಾಗಿರುವುದು)."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Describe symptoms"]

    # 1) extract symptom keys
    symptom_keys = _extract_symptom_keys(user_text, fuzzy_threshold=0.58)

    # 2) if none found, try splitting into clauses and match again (aggressive)
    if not symptom_keys:
        clauses = re.split(r"[,.;:/\\-]", user_text)
        for clause in clauses:
            keys = _extract_symptom_keys(clause, fuzzy_threshold=0.55)
            symptom_keys.extend(keys)

    # 3) dedupe
    symptom_keys = list(dict.fromkeys(symptom_keys))

    if not symptom_keys:
        fallback = {"en": "Couldn't identify clear symptoms. Please provide more details or upload a photo.",
                    "kn": "ನಿರ್ದಿಷ್ಟ ಲಕ್ಷಣಗಳು ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Contact Krishi Adhikari"]

    # 4) score candidate pests/diseases
    ranked = _score_candidates(symptom_keys, user_crop)

    if not ranked:
        fallback = {"en": "No candidate pests/diseases found for those symptoms.",
                    "kn": "ಆ ಲಕ್ಷಣಗಳಿಗೆ ಯೋಗ್ಯವಾದ ಕೀಟ/ರೋಗಗಳು ಕಂಡುಬರಲಿಲ್ಲ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Contact Krishi Adhikari"]

    # 5) build response text with top 3 candidates, confidence & evidence
    top_k = ranked[:3]
    lines = []
    if lang == "kn":
        header = "ಸರಾಸರಿ ಅನುಮಾನಿತ ರೋಗ/ಕೀಟಗಳು (ಮೇಲವರ್ಗ):\n"
    else:
        header = "Likely pests/diseases (top candidates):\n"
    lines.append(header)

    for cand, score, conf, ev in top_k:
        # friendly meta if available
        meta = DISEASE_META.get(cand, {})
        meta_note = meta.get("note", "")
        lines.append(f"- {cand.title()} (confidence: {int(conf*100)}%)")
        if meta_note:
            lines.append(f"    • {meta_note}")
        lines.append(f"    • Evidence: {', '.join(ev)}")

    # 6) suggestions and recommended actions: map to PESTICIDE_DB if available
    suggestions = ["Upload photo", "Contact Krishi Adhikari", "View prevention steps"]
    rec_texts = []
    for cand, score, conf, ev in top_k:
        # if an exact pesticide recommendation exists in PESTICIDE_DB for this key -> include it
        key = cand.lower()
        if key in PESTICIDE_DB:
            rec = PESTICIDE_DB[key].get(lang if lang in ["en", "kn"] else "en")
            if rec:
                rec_texts.append(f"For {cand.title()}: {rec}")

    if rec_texts:
        lines.append("\nSuggested interventions:")
        for r in rec_texts:
            lines.append(f"- {r}")
        suggestions.insert(0, "Pesticide recommendations")

    # optional: include the normalized symptom keys found
    lines.append("\nIdentified symptoms:")
    for s in symptom_keys:
        lines.append(f"- {s}")

    final_text = "\n".join(lines)
    return final_text, False, suggestions



# =========================================================
# NEW MODULE: Weather + crop stage fusion advisory
# - Combines stage recommendations + upcoming weather to give fused advice (e.g., skip fertilizer if rain predicted)
# =========================================================
def weather_crop_fusion(user_id: str, crop: str, stage: str, lang: str):
    farm = get_user_farm_details(user_id)
    district = farm.get("district", "unknown")

    weather = fetch_weather_by_location(district)
    if not weather:
        return ("Weather data unavailable.", False, ["Retry"])

    # Stage advice
    stage_advice = stage_recommendation_engine(crop, stage, lang)

    # Weather fusion suggestions
    fusion = weather_suggestion_engine(weather, crop_stage=stage, language=lang)

    # Build final message
    if lang == "kn":
        report = (
            f"{district} ಹವಾಮಾನ:\n"
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
# Gemini fallback (crop advisory)
# =========================================================
def get_prompt(lang: str) -> str:
    return f"You are KrishiSakhi. Respond only in {'Kannada' if lang == 'kn' else 'English'} with short actionable crop advice."


def crop_advisory(user_id: str, query: str, lang: str, session_key: str):
    global client, active_chats
    try:
        if not client:
            return "AI not configured on server.", False, [], session_key
        if session_key not in active_chats:
            cfg = types.GenerateContentConfig(system_instruction=get_prompt(lang))
            chat = client.chats.create(model="gemini-2.5-flash", config=cfg)
            active_chats[session_key] = chat
        chat = active_chats[session_key]
        resp = chat.send_message(query)
        text = resp.text if hasattr(resp, "text") else str(resp)
        return text, False, ["Crop stage", "Pest check", "Soil test"], session_key
    except Exception as e:
        return f"AI error: {e}", False, [], session_key


# =========================================================
# Crop stage retrieval (latest activity)
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
    # build response with stage-wise recommendation
    rec = stage_recommendation_engine(latest_crop, latest_stage, lang)
    if lang == "kn":
        header = f"{latest_crop} ಬೆಳೆ ಪ್ರಸ್ತುತ ಹಂತ: {latest_stage}\n\n"
    else:
        header = f"Current stage of {latest_crop}: {latest_stage}\n\n"
    return header + rec, False, ["Next actions", "Fertilizer advice", "Pest check"]
# =========================================================
# GENERAL AGRICULTURE KNOWLEDGE ENGINE
# =========================================================

GENERAL_AGRI_TOPICS = {
    "organic farming": {
        "en": "Organic farming avoids synthetic fertilizers and pesticides, using compost, FYM, crop rotation, biofertilizers and natural pest control to maintain soil health.",
        "kn": "ಜೈವಿಕ ಕೃಷಿಯಲ್ಲಿ ರಾಸಾಯನಿಕ ಗೊಬ್ಬರ/ವಿಷವಸ್ತುಗಳನ್ನು ತಪ್ಪಿಸಿ, ಕಂಪೋಸ್ಟ್, FYM, ಬೆಳೆ ಪರಿವರ್ತನೆ, ಜೈವಿಕ ಗೊಬ್ಬರ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕೀಟ ನಿಯಂತ್ರಣವನ್ನು ಬಳಸಿ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾಪಾಡುತ್ತಾರೆ."
    },
    "mulching": {
        "en": "Mulching covers soil with straw, leaves, plastic sheets etc. Benefits: moisture retention, weed control, reduced soil temperature, higher yield.",
        "kn": "ಮಲ್ಚಿಂಗ್ ಎಂದರೆ ಮಣ್ಣನ್ನು ಬಲುಸು, ಎಲೆ, ಪ್ಲಾಸ್ಟಿಕ್ ಶೀಟ್ ಇತ್ಯಾದಿಯಿಂದ ಮುಚ್ಚುವುದು. ಲಾಭಗಳು: ತೇವ ಉಳಿಕೆ, ಕಳೆ ನಿಯಂತ್ರಣ, ಮಣ್ಣಿನ ತಾಪಮಾನ ಕಡಿತ, ಹೆಚ್ಚಿನ ಉತ್ಪಾದನೆ."
    },
    "drip irrigation": {
        "en": "Drip irrigation delivers water directly to roots using pipes and emitters, reducing wastage and improving water-use efficiency by 40–60%.",
        "kn": "ಡ್ರಿಪ್ ನೀರಾವರಿ ಪೈಪು/ಇಮಿಟರ್ ಮೂಲಕ ನೀರನ್ನು ನೇರವಾಗಿ ಬೇರುಗಳಿಗೆ ಒದಗಿಸುತ್ತದೆ. 40–60% ನೀರು ಉಳಿಸುತ್ತದೆ."
    },
    "soil fertility": {
        "en": "Improve soil fertility with compost, green manure, crop rotation, earthworms, reduced chemical fertilizer use, and regular soil testing.",
        "kn": "ಕಂಪೋಸ್ಟ್, ಹಸಿರು ಗೊಬ್ಬರ, ಬೆಳೆ ಪರಿವರ್ತನೆ, ಮಣ್ಣು ಪರೀಕ್ಷೆ, ರಾಸಾಯನಿಕ ಗೊಬ್ಬರ ಕಡಿಮೆ ಬಳಕೆ — ಮಣ್ಣಿನ ಸುಭಿಕ್ಷತೆಗೆ ಮುಖ್ಯ."
    },
    "micronutrients": {
        "en": "Micronutrients (Zn, Fe, B, Mn, Cu, Mo) are required in small amounts but essential for crop growth. Deficiency causes yellowing, poor flowering, stunted growth.",
        "kn": "ಸುಕ್ಷಮ ಪೋಷಕಾಂಶಗಳು (Zn, Fe, B, Mn, Cu, Mo) ಕಡಿಮೆ ಪ್ರಮಾಣದಲ್ಲಿ ಬೇಕಾದರೂ ಬೆಳೆ ಬೆಳವಣಿಗೆಗೆ ಅಗತ್ಯ. ಕೊರತೆ → ಹಳದಿ ಎಲೆಗಳು, ಹೂ ಕುಗ್ಗುವುದು, ಬೆಳವಣಿಗೆ ತಡೆಯುವುದು."
    },
    "ipm": {
        "en": "Integrated Pest Management (IPM) uses biological, cultural, mechanical and limited chemical control to manage pests with minimal environmental impact.",
        "kn": "ಸಮಗ್ರ ಕೀಟ ನಿರ್ವಹಣೆ (IPM) → ಜೈವಿಕ, ಸಾಂಸ್ಕೃತಿಕ, ಯಾಂತ್ರಿಕ ಹಾಗೂ ಅಗತ್ಯವಿದ್ದರೆ ಮಾತ್ರ ರಾಸಾಯನಿಕ ಕ್ರಮಗಳನ್ನು ಬಳಸಿ ಕೀಟ ನಿಯಂತ್ರಣೆ."
    },
    "hybrid seed": {
        "en": "Hybrid seeds are produced by controlled pollination of two parent varieties. Benefits: higher yield, disease resistance, uniform growth.",
        "kn": "ಹೈಬ್ರಿಡ್ ಬೀಜಗಳನ್ನು ಎರಡು ಪ್ರಭೇದಗಳ ನಿಯಂತ್ರಿತ ಪರಾಗಸಂಚಯದಿಂದ ತಯಾರಿಸಲಾಗುತ್ತದೆ. ಲಾಭ: ಹೆಚ್ಚಿನ ಉತ್ಪಾದನೆ, ರೋಗನಿರೋಧಕತೆ, ಸಮಾನ ಬೆಳವಣಿಗೆ."
    },
    "composting": {
        "en": "Composting converts farm waste into nutrient-rich manure. Use layers of dry and green waste; keep moist; turn every 15 days.",
        "kn": "ಕಂಪೋಸ್ಟಿಂಗ್‌ನಲ್ಲಿ ಕೃಷಿ ತ್ಯಾಜ್ಯವನ್ನು ಪೋಷಕಾಂಶಯುಕ್ತ ಗೊಬ್ಬರವಾಗಿ ಪರಿವರ್ತಿಸಲಾಗುತ್ತದೆ. ಒಣ/ಹಸಿರು ಕಸ ಪದರಗಳನ್ನು ಬಳಸಿ; ತೇವ ಇರಲಿ; 15 ದಿನಗಳಲ್ಲಿ ಒಮ್ಮೆ ತಿರುಗಿಸಿರಿ."
    },
    "weed management": {
        "en": "Weed management includes mulching, shallow cultivation, hand weeding, crop rotation and selective herbicides.",
        "kn": "ಕಳೆ ನಿರ್ವಹಣೆಗೆ ಮಲ್ಚಿಂಗ್, ಮೇಲ್ಮೈ ಹೊಲಸುವುದು, ಕೈಯಿಂದ ಕಳೆ ತೆಗೆದುಹಾಕುವುದು, ಬೆಳೆ ಪರಿವರ್ತನೆ, ಆಯ್ಕೆಮಾಡಿದ ಹರಬ್ಬಿಸೈಡ್ ಬಳಸುವುದು."
    },
    "fertilizer types": {
        "en": "Fertilizers are of three types: chemical (NPK), organic (FYM, compost), biofertilizers (Azotobacter, Rhizobium).",
        "kn": "ಎರೆ ಮೂರು ವಿಧ: ರಾಸಾಯನಿಕ (NPK), ಜೈವಿಕ (FYM, ಕಂಪೋಸ್ಟ್), ಜೈವ ಗೊಬ್ಬರಗಳು (ಅಜೋಟೊಬ್ಯಾಕ್ಟರ್, ರೈಸೋಬಿಯಂ)."
    }
}


def general_agri_knowledge_engine(query: str, lang: str) -> Tuple[str, bool, list]:
    q = query.lower()

    # Keyword-based fuzzy detection
    for topic, info in GENERAL_AGRI_TOPICS.items():
        if topic in q:
            return info[lang], False, ["More details", "Related practices"]

    # Generic detection patterns
    general_keywords = [
        "what is", "how to", "advantages", "benefits", "best practice",
        "agriculture", "farming method", "soil health", "improve yield",
        "irrigation types", "organic", "fertility", "mulching", "compost"
    ]

    if any(k in q for k in general_keywords):
        # fallback: Gemini / your crop advisory model gives general info
        return (
            "General agriculture query detected. I can help with organic farming, soil health, irrigation, fertilizer types, IPM, mulching, composting, seed types and more. Please ask specifically.",
            False,
            ["Organic farming", "Soil health", "Irrigation", "IPM"]
        )

    return None, None, None


# =========================================================
# Router — identify intents and call modules
# =========================================================
def route(query: str, user_id: str, lang: str, session_key: str):
    q = query.lower().strip()

    # simple intent checks (order matters)
    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center"]):
        return {"response_text": soil_testing_center(user_id, lang)[0], "voice": True, "suggestions": ["Update farm details"]}

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
        t, v, s = pest_disease(query, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # fertilizer intent: user can say "fertilizer" or "how much fertilizer"
    if "fertilizer" in q or "fertiliser" in q or "apply fertilizer" in q:
        # try to pick crop & stage from user's farm (latest)
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None; latest_stage = None; latest_ts = -1
        for crop, entries in logs.items() if isinstance(logs, dict) else []:
            if isinstance(entries, dict):
                for aid, data in entries.items():
                    ts = data.get("timestamp", 0)
                    if ts and ts > latest_ts:
                        latest_ts = ts
                        latest_crop = data.get("cropName", crop)
                        latest_stage = data.get("stage", "")
        if not latest_crop:
            # fallback ask user to provide crop & stage
            msg = ("Please provide the crop and stage (e.g., 'fertilizer for paddy tillering')"
                   if lang == "en" else "ದಯವಿಟ್ಟು ಬೆಳೆ ಮತ್ತು ಹಂತವನ್ನು ನೀಡಿ (ಉದಾ: 'fertilizer for paddy tillering')")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = fertilizer_calculator(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # pesticide intent: user asks "what to spray for aphid" or "pesticide for fruit borer"
    if "pesticide" in q or "spray" in q or "aphid" in q or "fruit borer" in q:
        # try to extract pest name (simple)
        pest = None
        for key in PESTICIDE_DB.keys():
            if key in q:
                pest = key
                break
        if not pest:
            # fallback ask
            msg = ("Please tell me the pest name or upload a photo (e.g., 'aphid')." if lang == "en"
                   else "ದಯವಿಟ್ಟು ಕೀಟದ ಹೆಸರು ಅಥವಾ ಫೋಟೋ ನೀಡಿ (ಉದಾ: aphid).")
            return {"response_text": msg, "voice": False, "suggestions": ["Upload photo", "aphid"]}
        t, v, s = pesticide_recommendation("", pest, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # irrigation intent
    if "irrigation" in q or "water" in q or "irrigate" in q:
        # try latest crop & stage
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None; latest_stage = None; latest_ts = -1
        for crop, entries in logs.items() if isinstance(logs, dict) else []:
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

    # yield prediction intent
    if "yield" in q or "estimate" in q or "production" in q:
        # try to extract crop name; fallback to latest crop
        crop = None
        for c in BASE_YIELD_TON_PER_HA.keys():
            if c in q:
                crop = c
                break
        if not crop:
            logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
            latest_crop = None; latest_ts = -1
            for crop_k, entries in logs.items() if isinstance(logs, dict) else []:
                if isinstance(entries, dict):
                    for aid, data in entries.items():
                        ts = data.get("timestamp", 0)
                        if ts and ts > latest_ts:
                            latest_ts = ts
                            latest_crop = data.get("cropName", crop_k)
            crop = latest_crop or list(BASE_YIELD_TON_PER_HA.keys())[0]
        t, v, s = yield_prediction(crop, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}

    # weather + crop stage fusion intent
    if "weather" in q and "stage" in q:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        # extract latest crop/stage
        latest_crop = None; latest_stage = None; latest_ts = -1
        for crop, entries in logs.items() if isinstance(logs, dict) else []:
            if isinstance(entries, dict):
                for _, data in entries.items():
                    ts = data.get("timestamp", 0)
                    if ts > latest_ts:
                        latest_crop = data.get("cropName", crop)
                        latest_stage = data.get("stage", "")
        if not latest_crop:
            return {"response_text": "No crop found. Add crop activity.", "voice": False, "suggestions": ["Add activity"]}

        text, v, s = weather_crop_fusion(user_id, latest_crop, latest_stage, lang)
        return {"response_text": text, "voice": v, "suggestions": s}
        
        # ----------------------------------------------------------------------
    # WEATHER-BASED DISEASE PREDICTION
    # ----------------------------------------------------------------------
    if "disease" in q and "weather" in q:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None
        latest_ts = -1

        for crop, entries in logs.items() if isinstance(logs, dict) else []:
            if isinstance(entries, dict):
                for _, data in entries.items():
                    ts = data.get("timestamp", 0)
                    if ts > latest_ts:
                        latest_ts = ts
                        latest_crop = data.get("cropName", crop)

        if not latest_crop:
            return {
                "response_text": "No crop found. Add farm activity.",
                "voice": False,
                "suggestions": ["Add activity"]
            }

        farm = get_user_farm_details(user_id)
        weather = fetch_weather_by_location(farm.get("district", "unknown"))

        if not weather:
            return {
                "response_text": "Weather unavailable.",
                "voice": False,
                "suggestions": ["Retry"]
            }

        result = predict_disease_from_weather(latest_crop, weather, lang)
        return {
            "response_text": result,
            "voice": False,
            "suggestions": ["Pest check", "Preventive spray"]
        }

    # ----------------------------------------------------------------------
    # ADVANCED PEST/DISEASE DIAGNOSIS (SYMPTOM BASED)
    # ----------------------------------------------------------------------
    if any(tok in q for tok in ["pest", "disease", "symptom", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None
        latest_ts = -1

        for crop_k, entries in logs.items() if isinstance(logs, dict) else []:
            if isinstance(entries, dict):
                for aid, data in entries.items():
                    ts = data.get("timestamp", 0)
                    if ts > latest_ts:
                        latest_ts = ts
                        latest_crop = data.get("cropName", crop_k)

        diag_text, voice, sugg = diagnose_advanced(query, user_crop=latest_crop, lang=lang)
        return {
            "response_text": diag_text,
            "voice": voice,
            "suggestions": sugg
        }

    # General agriculture knowledge intent
    gen_text, gen_voice, gen_sugg = general_agri_knowledge_engine(query, lang)
    if gen_text:
        return {"response_text": gen_text, "voice": gen_voice, "suggestions": gen_sugg}


    # Default → Gemini crop advisory or fallback text
    t, v, s, sid = crop_advisory(user_id, query, lang, session_key)
    return {"response_text": t, "voice": v, "suggestions": s, "session_id": sid}


# =========================================================
# Endpoint
# =========================================================
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    user_query = payload.user_query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Detect user preferred language (en/kn)
    lang = get_language(payload.user_id)

    # Generate session key
    session_key = payload.session_id or f"{payload.user_id}-{lang}"

    # Route query through intent engine
    try:
        result = route(user_query, payload.user_id, lang, session_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    # -------------------------------------------------------------------------
    # TTS AUDIO GENERATION (Kannada or English)
    # -------------------------------------------------------------------------
    audio_url = None
    try:
        if result.get("response_text"):
            audio_url = generate_tts_audio(result["response_text"], lang)
    except Exception as e:
        print("TTS generation failed:", e)

    # -------------------------------------------------------------------------
    # Return unified response model
    # -------------------------------------------------------------------------
    return ChatResponse(
        session_id=result.get("session_id", session_key),
        response_text=result.get("response_text", "Sorry, could not process."),
        language=lang,
        suggestions=result.get("suggestions", []),
        voice=True,                    # always speak
        audio_url=audio_url,           # INCLUDE THE AUDIO URL
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )

# =========================================================
# Startup
# =========================================================
@app.on_event("startup")
def startup():
    initialize_firebase_credentials()
    initialize_gemini()

