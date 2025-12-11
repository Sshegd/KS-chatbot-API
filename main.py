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
import uuid
import traceback
import requests
import logging
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
from collections import defaultdict
from typing import Tuple
from fastapi.staticfiles import StaticFiles



# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
logger = logging.getLogger("ks-backend")
logger.setLevel(logging.INFO)
# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_API_KEY".upper()) or os.getenv("HF_API_KEY".lower())
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3-8B-Instruct")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY","")

if not FIREBASE_DATABASE_URL:
    raise Exception("FIREBASE_DATABASE_URL missing")
# Firebase scopes
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]
#global
credentials = None
app = FastAPI(title="KS Chatbot Backend", version="3.0")
# HuggingFace inference endpoints
HF_LLM_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
HF_API_URL_BASE = "https://api-inference.huggingface.co/models"
# TTS model
HF_TTS_URL = "https://api-inference.huggingface.co/models/sarvamai/sarvam-tts-multilingual"
# Active chat sessions
active_chats: Dict[str, List[Dict[str, str]]] = {}   # stores conversation history


# Ensure tts_audio dir exists before mounting
TTS_DIR = os.path.join(os.path.dirname(__file__), "tts_audio") if "__file__" in globals() else "./tts_audio"
os.makedirs(TTS_DIR, exist_ok=True)
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")


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

# =========================================================
# TTS generation (optional, gTTS fallback)
# =========================================================
def generate_tts_audio(text: str, lang: str):
    """
    Try to use gTTS. If gTTS is not installed or fails, return None (no audio).
    Returns URL path relative to static mount (e.g., /tts/tts_xxx.mp3) if saved successfully.
    """
    if not text:
        return None
    try:
        from gtts import gTTS
    except Exception:
        # gTTS not installed — skip TTS gracefully
        print("gTTS not installed; skipping TTS generation.")
        return None

    import uuid
    safe_lang = "kn" if lang == "kn" else "en"
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(TTS_DIR, filename)
    try:
        tts = gTTS(text=text, lang=safe_lang)
        tts.save(filepath)
        print("Saved TTS file:", filepath)
        return f"/tts/{filename}"
    except Exception as e:
        print("TTS generation error:", e)
        return None

# ============================================================
# FIREBASE TOKEN HANDLING
# ============================================================

def initialize_firebase_credentials():
    global credentials
    if credentials:
        return
    try:
        data = json.loads(SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(
            data, scopes=SCOPES
        )
        logger.info("Firebase credentials loaded.")
    except Exception as e:
        logger.error("Failed to load Firebase credentials: %s", e)
        raise

def get_firebase_token():
    global credentials
    if not credentials:
        initialize_firebase_credentials()
    try:
        if not credentials.token or credentials.expired:
            credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception as e:
        logger.error("Firebase token error: %s", e)
        raise 


def firebase_get(path: str):
    """GET helper for Firebase Realtime DB."""
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_DATABASE_URL}/{path}.json"
        res = requests.get(url, params={"access_token": token}, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error("Firebase GET error: %s", e)
        return None



# ============================================================
# USER LANGUAGE & FARM DETAILS FETCHERS
# ============================================================
def get_language(user_id: str) -> str:
    lang = firebase_get(f"Users/{user_id}/preferredLanguage")
    if isinstance(lang, str):
        return "kn" if lang.lower() == "kn" else "en"
    return "en"


def get_user_location(user_id: str):
    data = firebase_get(f"Users/{user_id}/farmDetails")
    if not isinstance(data, dict):
        return None
    if "district" in data and "taluk" in data:
        return {"district": data["district"], "taluk": data["taluk"]}
    return None


def get_user_farm_details(user_id: str):
    data = firebase_get(f"Users/{user_id}/farmDetails")
    return data if isinstance(data, dict) else {}


# ============================================================
# HELPER — CREATE TTS FOLDER ON STARTUP
# ============================================================
TTS_DIR = "tts_audio"
if not os.path.exists(TTS_DIR):
    os.makedirs(TTS_DIR)
    logger.info("Created /tts_audio directory.")

# expose static TTS audio files
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")


# ============================================================
# HELPERS - LANGUAGE TEXT SELECTOR
# ============================================================

def pick(text_en: str, text_kn: str, lang: str):
    return text_kn if lang == "kn" else text_en
# Helper to get latest crop & stage from farmActivityLogs defensively
def get_latest_crop_and_stage(user_id: str) -> Tuple[Optional[str], Optional[str]]:
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
    if not isinstance(logs, dict):
        return None, None
    latest_ts = -1
    latest_crop = None
    latest_stage = None
    for crop_key, entries in logs.items():
        if not isinstance(entries, dict):
            continue
        for act_id, data in entries.items():
            try:
                ts = int(data.get("timestamp", 0) or 0)
            except Exception:
                ts = 0
            if ts and ts > latest_ts:
                latest_ts = ts
                latest_crop = data.get("cropName") or crop_key
                latest_stage = data.get("stage", "Unknown")
    return latest_crop, latest_stage

# =====================================================
#knowledge base
# =====================================================
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

PRICE_LIST = {
    "paddy": 20,
    "chilli": 50,
    "ragi": 18,
    "arecanut": 470,
    "banana": 12,
    "turmeric": 120,
    "cotton": 40,
    "sugarcane": 3,      # per kg equivalent (₹3000/ton)

    # Cereals & Millets
    "maize": 18,
    "jowar": 25,
    "bajra": 22,
    "wheat": 24,
    "foxtail millet": 42,
    "little millet": 55,

    # Pulses
    "red gram": 110,
    "green gram": 95,
    "black gram": 80,
    "horse gram": 55,
    "cowpea": 60,

    # Oilseeds
    "groundnut": 55,
    "sunflower": 45,
    "soybean": 40,
    "sesame": 120,
    "castor": 50,

    # Fruits
    "mango": 25,
    "papaya": 10,
    "grapes": 35,
    "pomegranate": 90,
    "sapota": 20,

    # Vegetables
    "tomato": 12,
    "potato": 20,
    "onion": 18
}
# ===============================================================
# DISTRICT → LAT/LON MAPPING (Karnataka major districts)
# ===============================================================
DISTRICT_COORDS = {
    "uttara kannada": (14.8, 74.1),
    "udupi": (13.34, 74.74),
    "dakshina kannada": (12.87, 74.88),
    "shivamogga": (13.93, 75.56),
    "hassan": (13.01, 76.10),
    "kodagu": (12.34, 75.80),
    "mandya": (12.52, 76.90),
    "mysuru": (12.30, 76.65),
    "chamarajanagar": (11.93, 76.95),
    "bengaluru": (12.97, 77.59),
    "bengaluru rural": (13.19, 77.49),
    "ramanagara": (12.72, 77.27),
    "tumakuru": (13.34, 77.10),
    "chikkaballapur": (13.44, 77.72),
    "kolar": (13.13, 78.13),
    "chitradurga": (14.23, 76.40),
    "davangere": (14.47, 75.92),
    "ballari": (15.14, 76.92),
    "raichur": (16.21, 77.34),
    "koppal": (15.35, 76.15),
    "gadag": (15.43, 75.63),
    "haveri": (14.79, 75.40),
    "dharwad": (15.46, 75.01),
    "bidar": (17.91, 77.53),
    "kalaburagi": (17.33, 76.83),
    "yadgir": (16.75, 77.14),
    "belagavi": (15.85, 74.50),
    "vijayapura": (16.83, 75.71),
    "bagalkot": (16.18, 75.70)
}


# ---------------- helper text normalization & symptom matcher ----------------
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

def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
    scores = defaultdict(float)
    evidence = defaultdict(list)
    for sk in symptom_keys:
        mapped = SYMPTOM_DB.get(sk, [])
        for cand in mapped:
            base_weight = 1.0 + (0.25 if len(sk.split()) >= 2 else 0)
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
        ranked.append((cand, round(sc, 2), confidence, evidence.get(cand, [])))
    return ranked

# ===============================================================
# GET USER FARM DETAILS (Helper)
# ===============================================================
def get_user_farm_details(user_id: str):
    data = firebase_get(f"Users/{user_id}/farmDetails")
    if isinstance(data, dict):
        return data
    return {}

# =========================================================
# Domain functions (fertilizer calculator, pesticide, irrigation, yield, weather/advisory, diagnosis)
# =========================================================
# ===============================================================
# CROP STAGE ADVISORY ENGINE
# ===============================================================
def stage_recommendation_engine(crop: str, stage: str, lang: str):
    crop_l = crop.lower()
    stage_l = stage.lower()

    if crop_l in STAGE_RECOMMENDATIONS:
        for s, msg in STAGE_RECOMMENDATIONS[crop_l].items():
            if stage_l in s:
                return msg["kn"] if lang == "kn" else msg["en"]

    # Fallback Answer
    return ("No specific stage advisory available." if lang == "en"
            else "ಈ ಹಂತಕ್ಕೆ ವಿಶೇಷ ಸಲಹೆ ಲಭ್ಯವಿಲ್ಲ.")

# ===============================================================
# FERTILIZER CALCULATOR (Stage-wise N-P-K)
# ===============================================================
def fertilizer_calculator(crop: str, stage: str, lang: str):
    crop_l = crop.lower()
    stage_l = stage.lower()

    if crop_l in FERTILIZER_BASE:
        for st, (N, P, K) in FERTILIZER_BASE[crop_l].items():
            if stage_l in st:
                if lang == "kn":
                    return f"{crop} - {stage} ಹಂತ:\nN: {N}kg | P: {P}kg | K: {K}kg"
                return f"{crop} - {stage} stage:\nN: {N}kg | P: {P}kg | K: {K}kg"

    return ("Fertilizer data not available." if lang == "en"
            else "ರಸಗೊಬ್ಬರ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.")
# ===============================================================
# IRRIGATION SCHEDULE ENGINE
# ===============================================================
def irrigation_engine(crop: str, user_id: str, lang: str):
    crop_l = crop.lower()
    farm = get_user_farm_details(user_id)
    district = farm.get("district", "unknown")

    weather = fetch_weather_by_location(district)
    if not weather:
        weather = get_mock_weather_for_district(district)

    et = CROP_ET_BASE.get(crop_l, 4)

    # irrigation mm/day
    irrigation_mm = et - (weather["rain"] * 0.8)
    irrigation_mm = max(0, irrigation_mm)

    if lang == "kn":
        return (
            f"{crop} ನೀರಾವರಿ ಸಲಹೆ:\n"
            f"ET: {et} mm/day\n"
            f"ಮಳೆ: {weather['rain']} mm\n"
            f"ಇಂದು ಅಗತ್ಯ ನೀರಾವರಿ: {irrigation_mm:.1f} mm"
        )
    return (
        f"{crop} irrigation recommendation:\n"
        f"ET: {et} mm/day\n"
        f"Rain: {weather['rain']} mm\n"
        f"Required irrigation today: {irrigation_mm:.1f} mm"
    )
# ===============================================================
# MARKET PRICE ENGINE (30 crop support)
# ===============================================================
def market_price_engine(query: str, lang: str):
    q = query.lower()
    for crop, price in PRICE_LIST.items():
        if crop in q:
            if lang == "kn":
                return f"{crop} ಸರಾಸರಿ ಬೆಲೆ: ₹{price}/kg"
            return f"Average price of {crop}: ₹{price}/kg"

    if lang == "kn":
        return "ದಯವಿಟ್ಟು ಬೆಳೆ ಹೆಸರು ನೀಡಿ."
    return "Please specify the crop name."
# ===============================================================
# YIELD PREDICTION ENGINE
# ===============================================================
def yield_prediction_engine(crop: str, area_acres: float, lang: str):
    crop_l = crop.lower()

    if crop_l not in BASE_YIELD_TON_PER_HA:
        return ("Yield data not available." if lang == "en"
                else "ಉತ್ಪಾದನಾ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.")

    base_yield = BASE_YIELD_TON_PER_HA[crop_l]  # ton/ha

    # Convert acres → hectare
    area_ha = area_acres * 0.404

    predicted = base_yield * area_ha

    if lang == "kn":
        return f"ಅಂದಾಜು ಉತ್ಪಾದನೆ: {predicted:.2f} ಟನ್ (ಪ್ರದೇಶ: {area_acres} ಎಕರೆ)"

    return f"Estimated yield: {predicted:.2f} tons (Area: {area_acres} acres)"
# ===============================================================
# WEATHER DISEASE PREDICTION ENGINE
# ===============================================================
def disease_prediction_engine(crop: str, user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    district = farm.get("district", "unknown")
    crop_l = crop.lower()

    weather = fetch_weather_by_location(district)
    if not weather:
        return ("Weather data unavailable." if lang == "en"
                else "ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.")

    temp = weather["temp"]
    humidity = weather["humidity"]
    rain = weather["rain"]

    conditions = []
    if humidity > 80:
        conditions.append("high_humidity")
    if rain > 5:
        conditions.append("continuous_rain")
    if temp > 32 and humidity < 40:
        conditions.append("high_temp_low_humidity")

    risks = []
    for rule in DISEASE_WEATHER_RISK.get(crop_l, []):
        if rule["cond"] in conditions:
            risks.append(rule["disease"])

    if not risks:
        return ("No major disease risks detected." if lang == "en"
                else "ಪ್ರಮುಖ ರೋಗದ ಅಪಾಯ ಕಂಡುಬರಲಿಲ್ಲ.")

    if lang == "kn":
        return "ಸಾಧ್ಯವಾದ ರೋಗಗಳು:\n" + "\n".join(risks)

    return "Possible diseases:\n" + "\n".join(risks)
# ===============================================================
# ADVANCED PEST DIAGNOSIS ENGINE
# ===============================================================
def diagnose_pest(user_text: str, language: str):
    symptom_keys = match_symptoms(user_text)

    if not symptom_keys:
        fallback = {
            "en": "I could not identify the pest. Please describe more symptoms or upload a photo.",
            "kn": "ಕೀಟ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ಲಕ್ಷಣ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಕಳುಹಿಸಿ."
        }
        return fallback[language], ["Upload photo"]

    ranked = _score_candidates(symptom_keys)

    if not ranked:
        return fallback[language], ["Upload photo"]

    top_pest = ranked[0][0]

    # Now call pesticide recommendation
    pesticide_info = pesticide_recommendation_engine(top_pest, language)

    response = (
        f"Detected pest: {top_pest}\n\n"
        f"{pesticide_info}"
    )

    return response, ["More pest info", "Upload photo"]

# ===============================================================
# WEATHER ADVISORY ENGINE (Detailed)
# ===============================================================
def weather_advisory_engine(user_id: str, lang: str):
    farm = get_user_farm_details(user_id)
    district = farm.get("district", "unknown")

    weather = fetch_weather_by_location(district)

    if not weather:
        msg = {
            "en": "Weather information unavailable.",
            "kn": "ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."
        }
        return msg[lang], [], False

    # Basic suggestions (rain, heat, humidity, wind)
    suggestions = weather_suggestion_engine(weather, None, lang)

    # Add WEATHER CLASSIFICATION ENGINE output
    classified_conditions = classify_weather_condition(weather, lang)

    # Merge unique suggestions + conditions
    final_suggestions = list(dict.fromkeys(suggestions + classified_conditions))

    if lang == "kn":
        report = (
            f"🌤️ {district} ಹವಾಮಾನ ವರದಿ:\n"
            f"🌡️ ತಾಪಮಾನ: {weather['temp']}°C\n"
            f"💧 ತೇವಾಂಶ: {weather['humidity']}%\n"
            f"🌬️ ಗಾಳಿ: {weather['wind']} km/h\n"
            f"🌧️ ಮಳೆ: {weather['rain']} mm\n\n"
            f"➡️ ಹವಾಮಾನ ಆಧಾರಿತ ಸಲಹೆಗಳು ಕೆಳಗಿವೆ:"
        )
    else:
        report = (
            f"🌤️ Weather Report for {district}:\n"
            f"🌡️ Temperature: {weather['temp']}°C\n"
            f"💧 Humidity: {weather['humidity']}%\n"
            f"🌬️ Wind: {weather['wind']} km/h\n"
            f"🌧️ Rain: {weather['rain']} mm\n\n"
            f"➡️ Weather-based recommendations:"
        )

    return report, final_suggestions, True

# ===============================================================
# GENERAL AGRICULTURE KNOWLEDGE ENGINE
# ===============================================================
def general_agri_knowledge_engine(query: str, lang: str):
    q = query.lower()

    for topic, info in GENERAL_AGRI_TOPICS.items():
        if topic in q:
            return info[lang], False, ["More details", "Best Practices"]

    generic_keywords = [
        "what is", "how to", "benefit", "fertility", "compost", "soil health",
        "organic", "mulching", "irrigation", "farming"
    ]

    if any(k in q for k in generic_keywords):
        if lang == "kn":
            return (
                "ಸಾಮಾನ್ಯ ಕೃಷಿ ಪ್ರಶ್ನೆ ಪತ್ತೆಯಾಗಿದೆ. ಮಣ್ಣಿನ ಆರೋಗ್ಯ, ನೀರಾವರಿ, ಜೈವ ಗೊಬ್ಬರ, ಮಲ್ಚಿಂಗ್, ಕೀಟನಿಯಂತ್ರಣ ಮೊದಲಾದವುಗಳ ಬಗ್ಗೆ ಕೇಳಬಹುದು.",
                False,
                ["Organic farming", "Mulching", "Irrigation"]
            )
        return (
            "General agriculture query detected. Ask about soil health, irrigation, compost, fertilizers, pests, etc.",
            False,
            ["Organic farming", "Soil health", "Irrigation"]
        )

    return None, None, None
# ===============================================================
# PESTICIDE RECOMMENDATION ENGINE (Advanced)
# ===============================================================
def pesticide_recommendation_engine(pest: str, lang: str):
    pest_l = pest.lower()

    if pest_l not in PESTICIDE_DB:
        if lang == "kn":
            return "ಈ ಕೀಟಕ್ಕೆ ಶಿಫಾರಸು ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತಷ್ಟು ವಿವರ ನೀಡಿ."
        return "No pesticide recommendations found for this pest."

    data = PESTICIDE_DB[pest_l]

    if lang == "kn":
        return (
            f"ಜೈವ ನಿಯಂತ್ರಣ:\n{data['organic']['kn']}\n\n"
            f"ರಾಸಾಯನಿಕ ನಿಯಂತ್ರಣ:\n{data['chemical']['kn']}\n\n"
            "⚠ ಸುರಕ್ಷತಾ ಸೂಚನೆ: 5–7 ದಿನಗಳ ನಂತರ ಮಾತ್ರ ಕೊಯ್ಲು ಮಾಡಿ."
        )

    return (
        f"Organic control:\n{data['organic']['en']}\n\n"
        f"Chemical control:\n{data['chemical']['en']}\n\n"
        "⚠ Safety Note: Maintain 5–7 days PHI before harvest."
    )
import requests

# ===============================================================
# FETCH WEATHER BY LOCATION (Free API — No Key Needed)
# ===============================================================
def fetch_weather_by_location(district: str):
    try:
        district_l = district.lower().strip()

        if district_l not in DISTRICT_COORDS:
            print("Unknown district:", district)
            return None

        lat, lon = DISTRICT_COORDS[district_l]

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m,precipitation"
        )

        res = requests.get(url, timeout=8)
        data = res.json()

        if "current_weather" not in data:
            return None

        current = data["current_weather"]

        # Hourly rain & humidity fallback
        hourly = data.get("hourly", {})

        humidity = hourly.get("relativehumidity_2m", [60])[0]
        rain = hourly.get("precipitation", [0])[0]

        weather_info = {
            "temp": current.get("temperature", 28),
            "wind": current.get("windspeed", 5),
            "humidity": humidity,
            "rain": rain,
            "condition": current.get("weathercode", "Clear"),
            "description": _weather_code_to_text(current.get("weathercode", 0))
        }

        return weather_info

    except Exception as e:
        print("Weather fetch error:", e)
        return None
def _weather_code_to_text(code):
    mapping = {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Cloudy",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Snowfall",
        80: "Rain showers",
        95: "Thunderstorm"
    }
    return mapping.get(code, "Weather unknown")
# ===============================================================
# WEATHER CONDITION CLASSIFICATION ENGINE
# ===============================================================
def classify_weather_condition(weather: dict, lang: str):
    """
    Classifies weather into meaningful agricultural categories.
    Input weather dict must contain:
        temp, humidity, wind, rain, condition (optional)
    """

    temp = weather.get("temp", 30)
    humidity = weather.get("humidity", 60)
    wind = weather.get("wind", 5)
    rain = weather.get("rain", 0)
    cond = weather.get("condition", "").lower()

    categories = []

    # Temperature classification
    if temp >= 38:
        categories.append("extreme_heat")
    elif temp >= 32:
        categories.append("high_heat")
    elif temp <= 12:
        categories.append("cold_stress")

    # Humidity classification
    if humidity >= 85:
        categories.append("very_high_humidity")
    elif humidity >= 70:
        categories.append("high_humidity")
    elif humidity <= 30:
        categories.append("low_humidity")

    # Rain classification
    if rain >= 20:
        categories.append("heavy_rain")
    elif rain >= 5:
        categories.append("rainy")
    elif rain == 0:
        categories.append("dry")

    # Wind classification
    if wind >= 30:
        categories.append("storm_warning")
    elif wind >= 15:
        categories.append("wind_stress")

    # If nothing major → ideal weather
    if not categories:
        categories.append("ideal")

    # Language-specific descriptions
    messages_en = {
        "extreme_heat": "Extreme heat – risk of crop dehydration.",
        "high_heat": "High heat – increase irrigation frequency.",
        "cold_stress": "Cold stress – avoid fertilizer application.",
        "very_high_humidity": "Very high humidity – high fungal disease risk.",
        "high_humidity": "High humidity – increased fungal infection chance.",
        "low_humidity": "Low humidity – soil moisture loss likely.",
        "heavy_rain": "Heavy rainfall – avoid irrigation and spraying.",
        "rainy": "Rainy conditions – reduce irrigation.",
        "dry": "Dry weather – irrigation recommended.",
        "storm_warning": "Strong winds/storm – avoid spraying pesticides.",
        "wind_stress": "High wind – may cause lodging in crops.",
        "ideal": "Weather is ideal for farming operations."
    }

    messages_kn = {
        "extreme_heat": "ತೀವ್ರ ಬಿಸಿಲು – ಬೆಳೆ ಒಣಗುವ ಅಪಾಯ.",
        "high_heat": "ಹೆಚ್ಚು ಬಿಸಿಲು – ನೀರಾವರಿ ಪ್ರಮಾಣ ಹೆಚ್ಚಿಸಿ.",
        "cold_stress": "ತೀವ್ರ ಚಳಿ – ರಸಗೊಬ್ಬರ ಬಳಕೆ ತಪ್ಪಿಸಿ.",
        "very_high_humidity": "ಅತ್ಯಧಿಕ ತೇವಾಂಶ – ಫಂಗಸ್ ರೋಗದ ಅಪಾಯ ಹೆಚ್ಚು.",
        "high_humidity": "ಹೆಚ್ಚು ತೇವಾಂಶ – ಫಂಗಲ್ ಸೋಂಕಿನ ಸಾಧ್ಯತೆ.",
        "low_humidity": "ಕಡಿಮೆ ತೇವಾಂಶ – ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಡಿಮೆಯಾಗಬಹುದು.",
        "heavy_rain": "ಭಾರೀ ಮಳೆ – ನೀರಾವರಿ ಮತ್ತು ಸಿಂಪಡಣೆ ತಪ್ಪಿಸಿ.",
        "rainy": "ಮಳೆಯ ಹವಾಮಾನ – ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ.",
        "dry": "ಒಣ ಹವಾಮಾನ – ನೀರಾವರಿ ಅಗತ್ಯ.",
        "storm_warning": "ಬಲವಾದ ಗಾಳಿ/ಪ್ರಳಯ – ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ಬೇಡ.",
        "wind_stress": "ಬಲವಾದ ಗಾಳಿ – ಬೆಳೆ ಬೀಳುವ ಅಪಾಯ.",
        "ideal": "ಕೃಷಿಗೆ ಅನುಕೂಲಕರ ಹವಾಮಾನ."
    }

    translated = []
    for c in categories:
        translated.append(messages_kn[c] if lang == "kn" else messages_en[c])

    return translated
   
# ===============================================================
# ADVANCED DISEASE PREDICTION FROM WEATHER + CROP
# ===============================================================

def predict_disease_from_weather(user_id: str, crop: str, lang: str):
    crop_l = crop.lower()

    # Fetch farm location
    farm = get_user_farm_details(user_id)
    district = farm.get("district", None)

    if not district:
        msg = {
            "en": "Farm district not found. Update farm details.",
            "kn": "ಫಾರಂ ಜಿಲ್ಲೆಯ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. farmDetails ನವೀಕರಿಸಿ."
        }
        return msg[lang], False, ["Update farm details"]

    # Fetch weather
    weather = fetch_weather_by_location(district)
    if not weather:
        return (
            "Weather information unavailable." if lang == "en"
            else "ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.",
            False,
            ["Retry"]
        )

    temp = weather["temp"]
    humidity = weather["humidity"]
    rain = weather["rain"]
    wind = weather["wind"]
    cond = weather["condition"].lower()

    # Determine conditions
    detected_conditions = []

    if humidity > 80:
        detected_conditions.append("high_humidity")

    if rain > 10 or "heavy" in cond:
        detected_conditions.append("heavy_rain")

    if rain > 5:
        detected_conditions.append("continuous_rain")

    if temp > 32 and humidity < 40:
        detected_conditions.append("high_temp_low_humidity")

    if temp > 35:
        detected_conditions.append("high_temp")

    if "rain" in cond:
        detected_conditions.append("rainy")

    # Match diseases for this crop
    diseases = []
    for rule in DISEASE_WEATHER_RISK.get(crop_l, []):
        if rule["cond"] in detected_conditions:
            diseases.append(rule["disease"])

    # If no disease appears
    if not diseases:
        msg = {
            "en": f"No major disease risk detected for {crop}.",
            "kn": f"{crop} ಗಾಗಿ ಪ್ರಮುಖ ರೋಗದ ಅಪಾಯ ಕಂಡುಬರಲಿಲ್ಲ."
        }
        return msg[lang], False, ["Check pest symptoms", "View crop advisory"]

    # Build detailed report
    if lang == "kn":
        report = (
            f"{district} ಹವಾಮಾನ ಆಧರಿಸಿ {crop} ಬೆಳೆಗಾಗುವ ಸಾಧ್ಯ ರೋಗಗಳು:\n\n"
            f" ತಾಪಮಾನ: {temp}°C\n"
            f" ತೇವಾಂಶ: {humidity}%\n"
            f" ಮಳೆ: {rain} mm\n"
            f" ಗಾಳಿ: {wind} km/h\n\n"
            "⚠ ಸಾಧ್ಯವಾದ ರೋಗಗಳು:\n - " + "\n - ".join(diseases)
        )
    else:
        report = (
            f"Based on weather in {district}, the following diseases are likely for {crop}:\n\n"
            f" Temperature: {temp}°C\n"
            f" Humidity: {humidity}%\n"
            f" Rain: {rain} mm\n"
            f" Wind: {wind} km/h\n\n"
            "⚠ Possible Diseases:\n - " + "\n - ".join(diseases)
        )

    return report, True, ["Pest Check", "Fungicide Advice", "Crop Stage"]
# =====================================================================
# ADVANCED SYMPTOM DIAGNOSIS – Natural Language Pest/Disease Detection
# =====================================================================
# ------------------------------------------------------
# EXTRACT SYMPTOM KEYS FROM NATURAL LANGUAGE
# ------------------------------------------------------
def _extract_symptom_keys(text: str):
    text_l = text.lower()
    keys = []

    for symptom in SYMPTOM_DB.keys():
        # direct match
        if symptom in text_l:
            keys.append(symptom)
            continue

        # fuzzy match (for spelling mistakes)
        match = difflib.get_close_matches(symptom, [text_l], n=1, cutoff=0.6)
        if match:
            keys.append(symptom)

    return list(set(keys))
# ------------------------------------------------------
# SCORE POSSIBLE PEST/DISEASE CANDIDATES
# ------------------------------------------------------
def _score_candidates(symptom_keys: list, crop: str = None):
    scores = defaultdict(float)
    evidence = defaultdict(list)

    for sk in symptom_keys:
        candidates = SYMPTOM_DB.get(sk, [])
        for cand in candidates:
            scores[cand] += 1.0
            evidence[cand].append(f"matched symptom: {sk}")

    # Boost based on crop
    if crop:
        crop_l = crop.lower()
        if crop_l in CROP_SYMPTOM_WEIGHT:
            for cand, boost in CROP_SYMPTOM_WEIGHT[crop_l].items():
                scores[cand] += boost
                evidence[cand].append(f"crop relevance boost: {crop_l}")

    # Convert to list
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    results = []
    total = sum(scores.values()) if scores else 1

    for cand, sc in ranked:
        confidence = min(0.99, sc / (total + 1e-6))
        results.append({
            "condition": cand,
            "score": round(sc, 2),
            "confidence": round(confidence * 100, 1),
            "evidence": evidence[cand]
        })

    return results
# ------------------------------------------------------
# ADVANCED DIAGNOSIS MAIN FUNCTION
# ------------------------------------------------------
def diagnose_advanced_symptoms(user_text: str, crop: str, lang: str):
    symptom_keys = _extract_symptom_keys(user_text)

    if not symptom_keys:
        if lang == "kn":
            return (
                "ಲಕ್ಷಣಗಳಿಂದ ರೋಗ ಅಥವಾ ಕೀಟವನ್ನು ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ.",
                ["Upload photo", "Common pest symptoms"]
            )
        return (
            "Unable to identify disease/pest from symptoms. Please provide more details or upload a photo.",
            ["Upload photo", "Common pest symptoms"]
        )

    ranked = _score_candidates(symptom_keys, crop)

    # Choose top 1–3 conditions
    top = ranked[:3]

    if lang == "kn":
        msg = "ಲಕ್ಷಣಗಳ ಆಧಾರದ ಮೇಲೆ ಸಾಧ್ಯವಾದ ಸಮಸ್ಯೆಗಳು:\n"
        for r in top:
            msg += f"- {r['condition']} (ನಂಬಿಕೆ {r['confidence']}%)\n"
    else:
        msg = "Possible issues based on symptoms:\n"
        for r in top:
            msg += f"- {r['condition']} ({r['confidence']}% confidence)\n"

    suggestions = ["Pesticide recommendation", "Prevention tips", "Stage-wise advice"]

    return msg, suggestions






# =========================================================
# HF-backed crop advisory function (replaces Gemini)
# =========================================================
def get_prompt(lang: str) -> str:
    return f"You are KrishiSakhi. Respond concisely in {'Kannada' if lang == 'kn' else 'English'} with short actionable crop advice. Keep replies short and actionable."

def crop_advisory(user_id: str, query: str, lang: str, session_key: str):
    """
    Uses HF inference to generate crop-specific responses.
    Falls back to a helpful message if HF is not available.
    """
    try:
        # Compose context: include user's latest farm details if available (to improve generation)
        farm = get_user_farm_details(user_id) or {}
        farm_summary = ""
        if farm:
            parts = []
            for k in ("district", "soilType", "areaInHectares"):
                if farm.get(k):
                    parts.append(f"{k}:{farm.get(k)}")
            if parts:
                farm_summary = "Farm details: " + ", ".join(parts) + "\n\n"
        prompt = f"{get_prompt(lang)}\n\n{farm_summary}Farmer query: {query}\n\nGive short actionable steps."
        text, err = hf_generate_text(prompt, model=HF_MODEL, max_tokens=256, temperature=0.2)
        if err:
            # Log and provide fallback
            print("HF generation error:", err)
            fallback = {
                "en": "AI currently unavailable. I can still provide local rules: try asking for 'fertilizer', 'irrigation', 'pest' or 'soil test'.",
                "kn": "AI ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು 'fertilizer', 'irrigation', 'pest' ಅಥವಾ 'soil test' ಕೇಳಿ."
            }
            return fallback[lang], False, ["Fertilizer", "Irrigation", "Pest check"], session_key
        if not text:
            fallback = {"en": "No AI response generated.", "kn": "ಯಾವುದೇ ಉತ್ತರ ಸೃಷ್ಟಿಸಲಾಗಲಿಲ್ಲ."}
            return fallback[lang], False, ["Fertilizer", "Pest check"], session_key
        return text.strip(), False, ["Crop stage", "Pest check", "Soil test"], session_key
    except Exception as e:
        print("crop_advisory exception:", e)
        tb = traceback.format_exc()
        print(tb)
        fallback = {"en": "AI error occurred.", "kn": "AI ದೋಷ ಸಂಭವಿಸಿದೆ."}
        return fallback[lang], False, ["Fallback actions"], session_key

# =========================================================
# Router — identify intents and call modules
# =========================================================
def route(query: str, user_id: str, lang: str, session_key: str):
    q = query.lower().strip()
    # Intent checks (order matters)
    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center"]):
        text, voice, suggestions = ("Soil testing center lookup not implemented for all states. Update farm details.", True, ["Update farm details"])
        return {"response_text": text, "voice": voice, "suggestions": suggestions}
    if any(tok in q for tok in ["timeline", "activity log", "farm activity"]):
        # call farm_timeline (if implemented)
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
        if not logs:
            return {"response_text": "No activity logs found.", "voice": False, "suggestions": ["Add activity"]}
        # Build a quick summary
        summaries = []
        for crop, entries in (logs.items() if isinstance(logs, dict) else []):
            summaries.append(f"{crop}: {len(entries)} activities" if isinstance(entries, dict) else f"{crop}: activity")
        return {"response_text": "\n".join(summaries), "voice": False, "suggestions": ["View timeline"]}
    if any(tok in q for tok in ["weather", "rain", "forecast"]):
        report, sug, voice = weather_advisory(user_id, lang) if 'weather_advisory' in globals() else ("Weather module not configured.", [], False)
        return {"response_text": report, "voice": voice, "suggestions": sug}
    if any(tok in q for tok in ["price", "market", "mandi"]):
        t, v, s = market_price(query, lang) if 'market_price' in globals() else ("Market module not configured.", False, ["Ask price"])
        return {"response_text": t, "voice": v, "suggestions": s}
    if "crop stage" in q or q == "stage" or "stage" in q:
        t, v, s = get_latest_crop_stage(user_id, lang) if 'get_latest_crop_stage' in globals() else ("No crop stage module.", False, ["Add activity"])
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
        # For symptom heavy queries use advanced diagnosis
        diag_text, voice, sugg = diagnose_advanced(query, user_crop=None, lang=lang)
        return {"response_text": diag_text, "voice": voice, "suggestions": sugg}
    if "fertilizer" in q or "fertiliser" in q or "apply fertilizer" in q:
        # Try to detect crop & stage from farmActivityLogs
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
            msg = ("Please provide crop and stage (e.g., 'fertilizer for paddy tillering')" if lang == "en" else "ದಯವಿಟ್ಟು ಬೆಳೆ ಮತ್ತು ಹಂತ ನೀಡಿ.")
            return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
        t, v, s = fertilizer_calculator(latest_crop, latest_stage, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["pesticide", "spray", "aphid", "fruit borer"]):
        pest = None
        for key in PESTICIDE_DB.keys():
            if key in q:
                pest = key
                break
        if not pest:
            msg = ("Please tell me the pest name or upload a photo (e.g., 'aphid')." if lang == "en" else "ದಯವಿಟ್ಟು ಕೀಟದ ಹೆಸರು ಅಥವಾ ಫೋಟೋ ನೀಡಿ.")
            return {"response_text": msg, "voice": False, "suggestions": ["Upload photo", "aphid"]}
        t, v, s = pesticide_recommendation("", pest, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["irrigation", "water", "irrigate"]):
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
        for c in list(BASE_YIELD_TON_PER_HA.keys()):
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
            crop = latest_crop or list(BASE_YIELD_TON_PER_HA.keys())[0]
        t, v, s = yield_prediction(crop, user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    # default -> HF crop advisory
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
    lang = get_language(payload.user_id)
    session_key = payload.session_id or f"{payload.user_id}-{lang}"
    try:
        result = route(user_query, payload.user_id, lang, session_key)
    except Exception as e:
        print("Processing error:", e)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    audio_url = None
    try:
        if result.get("response_text"):
            audio_url = generate_tts_audio(result["response_text"], lang)
    except Exception as e:
        print("TTS generation failed:", e)
    return ChatResponse(
        session_id=result.get("session_id", session_key),
        response_text=result.get("response_text", "Sorry, could not process."),
        language=lang,
        suggestions=result.get("suggestions", []),
        voice=True,
        audio_url=audio_url,
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )

# =========================
# Additional helper stubs (market_price, weather_advisory, get_latest_crop_stage)
# You can replace/extend these with full versions from your earlier file.
# =========================

def get_latest_crop_stage(user_id: str):
    """
    Reads Firebase farmActivityLogs and returns:
    {
        "crop": "paddy",
        "stage": "tillering",
        "timestamp": 1712345678
    }
    Returns None if no stage exists.
    """

    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs or not isinstance(logs, dict):
        return None

    latest_crop = None
    latest_stage = None
    latest_ts = -1

    # logs: { "paddy": { "logId1": {...}, "logId2": {...} }, "chilli": {...} }
    for crop, entries in logs.items():
        if not isinstance(entries, dict):
            continue

        for _, entry in entries.items():
            if not isinstance(entry, dict):
                continue

            ts = entry.get("timestamp")
            stage = entry.get("stage")

            if ts and stage:
                if ts > latest_ts:
                    latest_ts = ts
                    latest_crop = entry.get("cropName", crop)
                    latest_stage = stage

    if latest_crop and latest_stage:
        return {
            "crop": latest_crop.lower(),
            "stage": latest_stage.lower(),
            "timestamp": latest_ts
        }

    return None

def get_user_location(user_id: str):
    """
    Returns user's district & taluk from Firebase:
    Path: Users/{user_id}/farmDetails
    Returns:
        { "district": "...", "taluk": "..." }
    or None if not available.
    """

    farm = firebase_get(f"Users/{user_id}/farmDetails")
    if not farm or not isinstance(farm, dict):
        return None

    district = farm.get("district")
    taluk = farm.get("taluk")

    if not district or not taluk:
        return None

    return {
        "district": district,
        "taluk": taluk
    }

def soil_testing_center(user_id: str, language: str):
    """
    Fetch nearest soil testing center based on user's district & taluk.
    Path: SoilTestingCenters/Karnataka/{district}/{taluk}
    """

    # Load user's saved farm location
    loc = get_user_location(user_id)
    if not loc:
        msg = {
            "en": "Farm location not found. Please update your district and taluk in farm details.",
            "kn": "ಫಾರಂ ಸ್ಥಳದ ಮಾಹಿತಿ ಕಂಡುಬರಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಜಿಲ್ಲೆ ಮತ್ತು ತಾಲೂಕು farmDetails ನಲ್ಲಿ ನವೀಕರಿಸಿ."
        }
        return msg[language], True, ["Update farm details"]

    district = loc.get("district")
    taluk = loc.get("taluk")

    if not district or not taluk:
        msg = {
            "en": "District or taluk missing in your farm details.",
            "kn": "ಫಾರಂ ವಿವರಗಳಲ್ಲಿ ಜಿಲ್ಲೆ ಅಥವಾ ತಾಲೂಕು ಲಭ್ಯವಿಲ್ಲ."
        }
        return msg[language], True, ["Update farm details"]

    # Firebase read
    centers = firebase_get(f"SoilTestingCenters/Karnataka/{district}/{taluk}")

    if not centers:
        msg = {
            "en": f"No soil testing center found for {taluk}, {district}.",
            "kn": f"{district} ಜಿಲ್ಲೆಯ {taluk} ತಾಲೂಕಿನ ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಕೇಂದ್ರ ಲಭ್ಯವಿಲ್ಲ."
        }
        return msg[language], True, ["Update farm details"]

    # Extract center information
    for _, info in centers.items():
        if isinstance(info, dict):
            name = info.get("name", "N/A")
            address = info.get("address", "N/A")
            contact = info.get("contact", "N/A")

            if language == "kn":
                text = (
                    f"🧪 ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಕೇಂದ್ರ:\n"
                    f"{name}\n\n"
                    f"📍 ವಿಳಾಸ: {address}\n"
                    f"📞 ಸಂಪರ್ಕ: {contact}"
                )
            else:
                text = (
                    f"🧪 Soil Testing Center:\n"
                    f"{name}\n\n"
                    f"📍 Address: {address}\n"
                    f"📞 Contact: {contact}"
                )

            return text, True, ["Directions", "Call center"]

    # Fallback
    no_data = {
        "en": "No center data available.",
        "kn": "ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಕೇಂದ್ರದ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."
    }
    return no_data[language], True, []

def pest_disease(query: str, language: str):
    q = query.lower().strip()

    # ---------------------------------------------------------
    # Keyword-based fast symptom match
    # ---------------------------------------------------------
    SYMPTOM_MAP = {
        "curl": {
            "disease_en": "Leaf curl virus or sucking pests (whiteflies/aphids).",
            "disease_kn": "ಎಲೆ ಕರ್ಭಟ ವೈರಸ್ ಅಥವಾ ಸ್ಯಕ್ಕಿಂಗ್ ಕೀಟಗಳು (ವೈಟ್‌ಫ್ಲೈ/ಆಫಿಡ್).",
            "advice_en": "Remove affected shoots and spray 2% neem oil or imidacloprid (as per label).",
            "advice_kn": "ಸೋಂಕಿತ ಕೊಂಬೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು 2% ನೀಮ್ ಎಣ್ಣೆ ಅಥವಾ ಲೇಬಲ್ ಪ್ರಕಾರ ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಸಿಂಪಡಿಸಿ.",
            "suggestions": ["Neem spray", "Pest control guide"]
        },

        "yellow": {
            "disease_en": "Likely nutrient deficiency (Nitrogen/Iron) or overwatering.",
            "disease_kn": "ಪೋಷಕಾಂಶ ಕೊರತೆ (ನೈಟ್ರೋಜನ್/ಐರನ್) ಅಥವಾ ಹೆಚ್ಚಾದ ನೀರಾವರಿ.",
            "advice_en": "Check soil moisture, reduce watering, apply urea or micronutrient mixture.",
            "advice_kn": "ಮಣ್ಣಿನ ತೇವಾಂಶ ಪರಿಶೀಲಿಸಿ, ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ, ಯೂರಿಯಾ ಅಥವಾ ಸೂಕ್ಷ್ಮಾಂಶ ಮಿಶ್ರಣ ನೀಡಿ.",
            "suggestions": ["Soil test", "Nutrient guide"]
        },

        "spots": {
            "disease_en": "Leaf spots indicate fungal disease (Anthracnose / Cercospora).",
            "disease_kn": "ಎಲೆಗಳಲ್ಲಿ ಕಲೆಗಳು ಫಂಗಲ್ ರೋಗ (ಆಂಥ್ರಾಕ್ನೋಸ್ / ಸರ್ಸ್ಪೋರಾ) ಸೂಚನೆ.",
            "advice_en": "Remove infected leaves and spray a recommended fungicide.",
            "advice_kn": "ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಶಿಫಾರಸು ಮಾಡಿದ ಫಂಗಿಸೈಡ್ ಸಿಂಪಡಿಸಿ.",
            "suggestions": ["Fungicide list", "Remove infected leaves"]
        },

        "brown": {
            "disease_en": "Brown patches suggest blight or leaf spot disease.",
            "disease_kn": "ಬ್ರೌನ್ ಕಲೆಗಳು ಬ್ಲೈಟ್ ಅಥವಾ ಎಲೆ ರೋಗದ ಸೂಚನೆ.",
            "advice_en": "Improve drainage and spray copper oxychloride.",
            "advice_kn": "ನೀರು ನಿಕಾಸ ಸುಧಾರಿಸಿ ಮತ್ತು ಕಾಪರ್ ಆಕ್ಸಿ ಕ್ಲೋರೈಡ್ ಸಿಂಪಡಿಸಿ.",
            "suggestions": ["Blight treatment", "Drainage tips"]
        },

        "wilt": {
            "disease_en": "Possible wilt (Fusarium/Bacterial).",
            "disease_kn": "ವಿಲ್ಟ್ ರೋಗ (ಫ್ಯೂಸೇರಿಯಮ್/ಬ್ಯಾಕ್ಟೀರಿಯಲ್) ಸಾಧ್ಯತೆ.",
            "advice_en": "Ensure good drainage, apply Trichoderma around roots.",
            "advice_kn": "ನೀರು ನಿಕಾಸ ಸುಧಾರಿಸಿ, ಬೇರುಗಳ ಬಳಿ ಟ್ರೈಕೋಡರ್ಮಾ ಬಳಸಿ.",
            "suggestions": ["Root treatment", "Soil solarization"]
        },

        "holes": {
            "disease_en": "Leaf holes indicate caterpillar or leaf-eating insects.",
            "disease_kn": "ಎಲೆಗಳಲ್ಲಿ ರಂಧ್ರಗಳು ಇರುವುದು ಹುಳು / ಎಲೆ ತಿನ್ನುವ ಕೀಟಗಳ ಲಕ್ಷಣ.",
            "advice_en": "Use pheromone traps and spray neem oil.",
            "advice_kn": "ಫೆರೊಮೋನ್ ಟ್ರ್ಯಾಪ್‌ಗಳು ಮತ್ತು ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ.",
            "suggestions": ["Pheromone traps", "Caterpillar management"]
        },

        "white powder": {
            "disease_en": "Powdery mildew detected.",
            "disease_kn": "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ರೋಗ ಕಂಡುಬಂದಿದೆ.",
            "advice_en": "Spray wettable sulfur or recommended fungicide.",
            "advice_kn": "ವೆಟ್ಟಬಲ್ ಸಲ್ಫರ್ ಅಥವಾ ಶಿಫಾರಸು ಮಾಡಿದ ಫಂಗಿಸೈಡ್ ಸಿಂಪಡಿಸಿ.",
            "suggestions": ["Sulphur spray", "Humidity control"]
        },

        "black spots": {
            "disease_en": "Black spots indicate fungal or bacterial infection.",
            "disease_kn": "ಕಪ್ಪು ಕಲೆಗಳು ಫಂಗಲ್ ಅಥವಾ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ರೋಗ.",
            "advice_en": "Remove affected leaves and avoid overhead irrigation.",
            "advice_kn": "ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಮೇಲಿನಿಂದ ನೀರು ಎರೆಯುವುದನ್ನು ತಪ್ಪಿಸಿ.",
            "suggestions": ["Disease control", "Irrigation tips"]
        },

        "sticky": {
            "disease_en": "Sticky leaves indicate honeydew from sucking pests (aphids/whiteflies).",
            "disease_kn": "ಎಲೆಗಳು ಅಂಟಿಕೊಂಡಿರುವುದು ವೈಟ್‌ಫ್ಲೈ/ಆಫಿಡ್ ಕೀಟದ ಹನಿ.",
            "advice_en": "Spray neem oil or soap solution.",
            "advice_kn": "ನೀಮ್ ಎಣ್ಣೆ ಅಥವಾ ಸಾಬೂನು ದ್ರಾವಣ ಸಿಂಪಡಿಸಿ.",
            "suggestions": ["Neem spray", "IPM method"]
        }
    }

    # ---------------------------------------------------------
    # Check for symptom patterns
    # ---------------------------------------------------------
    for symptom, data in SYMPTOM_MAP.items():
        if symptom in q:
            if language == "kn":
                text = f"{data['disease_kn']}\n\n➡ ಪರಿಹಾರ:\n{data['advice_kn']}"
            else:
                text = f"{data['disease_en']}\n\n➡ Solution:\n{data['advice_en']}"

            return text, True, data["suggestions"]

    # ---------------------------------------------------------
    # No direct match → fallback generic response
    # ---------------------------------------------------------
    fallback = {
        "en": "I could not identify the issue clearly. Please provide more details or upload a photo.",
        "kn": "ಸಮಸ್ಯೆಯನ್ನು ನಿಖರವಾಗಿ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."
    }

    return fallback[language], True, ["Upload photo", "Show common symptoms"]

def farm_timeline(user_id: str, language: str):
    """
    Fetches all activity logs from Firebase for the farmer and returns a summary  
    of the latest activity for each crop.

    Returns:
        (text, voice_flag, suggestions_list)
    """
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")

    if not logs or not isinstance(logs, dict):
        msg = "ಚಟುವಟಿಕೆ ಲಾಗ್ ಕಂಡುಬರಲಿಲ್ಲ." if language == "kn" else "No activity logs found."
        return msg, False, ["Add activity"]

    summaries = []

    for crop, entries in logs.items():

        if not isinstance(entries, dict):
            continue

        latest_entry = None
        latest_ts = -1

        # Find latest timestamp for this crop
        for act_id, data in entries.items():
            if not isinstance(data, dict):
                continue

            ts = data.get("timestamp", 0)
            if ts and ts > latest_ts:
                latest_ts = ts
                latest_entry = data

        if latest_entry:
            crop_name = latest_entry.get("cropName", crop)
            activity = latest_entry.get("subActivity", "")
            stage = latest_entry.get("stage", "")

            if language == "kn":
                summaries.append(f"{crop_name}: ಇತ್ತೀಚಿನ ಚಟುವಟಿಕೆ {activity} (ಹಂತ: {stage})")
            else:
                summaries.append(f"{crop_name}: latest activity {activity} (stage: {stage})")

    if not summaries:
        msg = "ಯಾವುದೇ ಇತ್ತೀಚಿನ ಚಟುವಟಿಕೆಗಳು ಕಂಡುಬರಲಿಲ್ಲ." if language == "kn" else "No recent activities found."
        return msg, False, ["Add activity"]

    # Final summary
    timeline_text = "\n".join(summaries)
    return timeline_text, False, ["View full timeline"]

def get_mock_weather_for_district(district):
    # Simple fallback mock weather (used if live fetch fails in irrigation schedule)
    return {
        "temp": 30,
        "humidity": 70,
        "wind": 8,
        "rain_next_24h_mm": 0
    }

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

def match_symptoms(text):
    return _extract_symptom_keys(text)

def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
    from collections import defaultdict
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

def crop_advisory(user_id: str, query: str, lang: str, session_key: str):
    global gemini_client, active_chats
    try:
        if not gemini_client:
            return "AI not configured on server.", False, [], session_key
        if session_key not in active_chats:
            if types is None:
                return "AI configuration incomplete.", False, [], session_key
            cfg = types.GenerateContentConfig(system_instruction=get_prompt(lang))
            chat = gemini_client.chats.create(model="gemini-1.5-flash", config=cfg)
            active_chats[session_key] = chat
        chat = active_chats[session_key]
        resp = chat.send_message(query)
        text = resp.text if hasattr(resp, "text") else str(resp)
        return text, False, ["Crop stage", "Pest check", "Soil test"], session_key

            
    except Exception as e:
        logger.exception("AI error: %s", e)
        return f"AI error: {e}", False, [], session_key
        
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
# HUGGINGFACE FALLBACK LLM (ZEHPYR 7B)
# =========================================================

import requests

HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def hf_generate(prompt: str) -> str:
    """
    Lightweight HF Inference API call.
    """
    try:
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 180}}

        r = requests.post(url, headers=headers, json=payload, timeout=45)

        if r.status_code != 200:
            return f"HF Error: {r.text}"

        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        return str(data)

    except Exception as e:
        return f"HF failure: {e}"

def route_intent(query: str, user_id: str, lang: str, session_key: str):
    q = query.lower()

    # 1. Soil test center
    if any(x in q for x in ["soil test", "soil center", "testing center", "soil lab"]):
        text, voice, sug = soil_testing_center(user_id, lang)
        return text, sug, True, session_key

    # 2. Farm timeline
    if "timeline" in q or "activity" in q:
        text, voice, sug = farm_timeline(user_id, lang)
        return text, sug, voice, session_key

    # 3. Weather advisory (full weather report)
    if "weather" in q or "rain" in q or "forecast" in q:
        text, sug, voice = weather_advisory(user_id, lang)
        return text, sug, voice, session_key

    # 4. Pest / disease direct keywords
    if any(x in q for x in ["pest", "disease", "spot", "curl", "yellow", "larva"]):
        text, sug = diagnose_pest(query, lang)
        return text, sug, True, session_key

    # 5. Market price
    if "price" in q or "mandi" in q or "rate" in q:
        text, voice, sug = market_price(query, lang)
        return text, sug, voice, session_key

    # 6. Stage-wise recommendation
    if "stage" in q or "growth" in q:
        crop, stage = get_latest_crop_stage(user_id)
        if crop:
            text = stage_recommendation_engine(crop, stage, lang)
            return text, ["Fertilizer", "Pest check", "Irrigation"], True, session_key
        return ("No crop stage found." if lang == "en" else "ಬೆಳೆಯ ಹಂತ ದೊರಕಲಿಲ್ಲ."), [], False, session_key

    # 7. General agri-knowledge
    ga, voice, sug = general_agri_knowledge_engine(query, lang)
    if ga:
        return ga, sug, voice, session_key

    # 8. Default → HuggingFace LLM
    prompt = f"You are KrishiSakhi, an agriculture assistant. Respond in {lang}. {query}"
    text = hf_generate(prompt)
    return text, ["Crop stage", "Pest check", "Soil test"], True, session_key

from gtts import gTTS
import uuid

def generate_tts(text: str, lang: str):
    """
    Always generate an mp3 file for the response (TTS-A mode).
    """
    try:
        audio_id = f"tts_{uuid.uuid4().hex}.mp3"
        path = f"tts_audio/{audio_id}"

        tts_lang = "kn" if lang == "kn" else "en"

        tts = gTTS(text=text, lang=tts_lang)
        tts.save(path)

        return f"/tts/{audio_id}"

    except Exception as e:
        logger.error("TTS error: %s", e)
        return None

@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    if not payload.user_query or not payload.user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    lang = get_language(payload.user_id)
    session_key = payload.session_id or f"{payload.user_id}-{lang}"

    try:
        text, suggestions, voice, sid = route_intent(
            payload.user_query.strip(),
            payload.user_id,
            lang,
            session_key
        )

    except Exception as e:
        logger.exception("Routing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Routing failure: {e}")

    # Always generate TTS (TTS-A)
    audio_url = generate_tts(text, lang)

    return ChatResponse(
        session_id=sid,
        response_text=text,
        language=lang,
        suggestions=suggestions,
        voice=True,               # Always speak
        metadata={
            "timestamp": datetime.utcnow().isoformat(),
            "audio_url": audio_url
        }
    )

# =========================================================
# STARTUP INITIALIZATION
# =========================================================
@app.on_event("startup")
def startup_event():
    logger.info("🔵 KS Backend Starting Up...")

    # ---------------------------
    # Ensure TTS folder exists
    # ---------------------------
    if not os.path.exists("tts_audio"):
        os.makedirs("tts_audio", exist_ok=True)
        logger.info("📁 Created tts_audio directory")

    # ---------------------------
    # Initialize Firebase
    # ---------------------------
    try:
        initialize_firebase_credentials()
        logger.info("✅ Firebase credentials loaded")
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")

    # ---------------------------
    # Initialize HuggingFace
    # ---------------------------
    global HF_API_KEY
    if HF_API_KEY:
        logger.info("✅ HuggingFace API key detected")
    else:
        logger.warning("⚠ HuggingFace API key missing! AI fallback may fail.")

    logger.info("🚀 KS Backend Startup Complete")




