import os
import json
import uuid
import requests
import traceback
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
import re
import difflib
import logging

# ---- load .env ----
load_dotenv()

# ---- Environment ----
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
# IMPORTANT: SERVICE_ACCOUNT_KEY must be the JSON file content as a single environment variable string.
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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
    logger.warning("gTTS not available: %s. Install via `pip install gTTS`", e)
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
        # Load service account info from the JSON string environment variable
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
        # Refresh the token if needed
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
        # Use access_token for server-side auth
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

# Mount static TTS dir so the client can access the generated audio files
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")

# ---- Pydantic Models ----
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
# Agriculture Data & Diagnosis (Partial data included for completeness)
# -------------------------

PRICE_LIST = {
    "chilli": 50, "paddy": 20, "ragi": 18, "turmeric": 120, "cotton": 40, 
    "tomato": 10, "onion": 25, "coffee": 300
}
FERTILIZER_BASE = {
    "paddy": {"nursery": (20,10,10), "tillering": (60,30,20)},
    "maize": {"vegetative": (80,40,20)}
}
PESTICIDE_DB = {
    "aphid": {"en": "Spray neem oil (2%) or insecticidal soap.", "kn": "ನೀಮ್ ಎಣ್ಣೆ (2%) ಸಿಂಪಡಿಸಿ."},
    "whitefly": {"en": "Use yellow sticky traps, neem oil (2%).", "kn": "ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್, ನೀಮ್ ಎಣ್ಣೆ (2%)."},
}
BASE_YIELD_TON_PER_HA = {
    "paddy": 4.0, "maize": 3.5, "ragi": 1.8
}
STAGE_RECOMMENDATIONS = {
    "paddy": {
        "tillering": {"en": "Apply urea (N); maintain 3–5 cm water; manage weeds.", "kn": "ಯೂರಿಯಾ (N) ನೀಡಿ; 3–5 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಗಿಡ್ಮುಳ್ಳು ನಿಯಂತ್ರಿಸಿ."},
    },
}
SYMPTOM_DB = {
    "yellow leaves": ["nutrient deficiency", "leaf curl virus"],
    "leaf curling": ["leaf curl virus", "thrips"],
}
DISEASE_META = {
    "leaf curl virus": {"type": "viral", "note": "Usually transmitted by whiteflies"},
}
CROP_SYMPTOM_WEIGHT = {"paddy": {"blast": 1.8}, "tomato": {"late blight": 2.0}}
SYMPTOM_SYNONYMS = {
    "yellowing": "yellow leaves",
}
# Placeholder functions for the router (must be fully defined in your final production code)
def _normalize_text(text: str) -> str: return text.lower()
def _extract_symptom_keys(user_text: str, fuzzy_threshold: float = 0.6): return []
def _score_candidates(symptom_keys: list, crop: Optional[str] = None): return []
def get_latest_crop_stage(user_id: str, lang: str): 
    return ("No stage data." if lang=="en" else "ಹಂತ ಲಭ್ಯವಿಲ್ಲ.", False, [])
def stage_recommendation_engine(crop_name: str, stage: str, lang: str): 
    return STAGE_RECOMMENDATIONS.get(crop_name.lower(), {}).get(stage.lower(), {}).get(lang, "No rec.")
def pesticide_recommendation(crop: str, pest: str, lang: str):
    p = PESTICIDE_DB.get(pest.lower(), {})
    return p.get(lang, f"Rec for {pest}"), False, ["Check photo"]
def market_price(query: str, language: str):
    q = (query or "").lower()
    for crop, price in PRICE_LIST.items():
        if crop in q:
            if language == "kn":
                return f"{crop.title()} ಸರಾಸರಿ ಬೆಲೆ: ₹{price}/ಕಿ.ಗ್ರಾಂ. ಸ್ಥಳೀಯ APMC ಪರಿಶೀಲಿಸಿ.", False, ["Sell"]
            return f"Approx price for {crop.title()}: ₹{price}/kg. Verify with local APMC.", False, ["Sell"]
    return {"en": "Specify crop.", "kn": "ಬೆಳೆ ನೀಡಿ"}[language], False, []
def fetch_weather_by_location(district: str): return {"temp": 30, "humidity": 70, "wind": 8, "rain": 0, "condition": "Clear", "description": "clear sky"}
def get_mock_weather_for_district(district): return {"temp": 30, "humidity": 70, "wind": 8, "rain": 0, "condition": "Clear", "description": "clear sky"}
def weather_suggestion_engine(weather, crop_stage=None, language="en"): return ["Irrigation recommended."]
def weather_advisory(user_id: str, language: str): 
    weather = fetch_weather_by_location("Bangalore") 
    report = f"Weather in Bangalore: {weather.get('temp')}°C."
    return report, weather_suggestion_engine(weather), True
def soil_testing_center(user_id: str, language: str): return ("No center data.", True, [])
def farm_timeline(user_id: str, language: str): return ("No logs found.", False, ["Add activity"])
def pest_disease(query: str, language: str): return ({"en": "Need details.", "kn": "ವಿವರ ಬೇಕು"}[language], True, [])
def fertilizer_calculator(crop: str, stage: str, user_id: str, lang: str): return ({"en": "Rec.", "kn": "ಸಲಹೆ"}[lang], False, ["Soil test"])
def irrigation_schedule(crop: str, stage: str, user_id: str, lang: str): return ({"en": "Rec.", "kn": "ಸಲಹೆ"}[lang], False, ["Soil moisture"])
def yield_prediction(crop: str, user_id: str, lang: str): return ({"en": "Estimate.", "kn": "ಅಂದಾಜು"}[lang], False, ["Improve"])
def diagnose_advanced(user_text: str, user_crop: Optional[str] = None, lang: str = "en"): return ({"en": "Diagnosis.", "kn": "ನಿರ್ಣಯ"}[lang], False, ["Photo"])
# -------------------------

# -------------------------
# TTS generation (gTTS)
# -------------------------
def generate_tts_audio(text: str, lang: str) -> Optional[str]:
    if not GTTS_AVAILABLE:
        logger.warning("gTTS not available.")
        return None
    try:
        code = "kn" if lang == "kn" else "en"
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(TTS_DIR, filename)
        if len(text) > 4000: text = text[:4000] # gTTS limitation
        tts = gTTS(text=text, lang=code)
        tts.save(filepath)
        return f"/tts/{filename}"
    except Exception as e:
        logger.exception("TTS error: %s", e)
        return None

# -------------------------
# Hugging Face Router client for text generation (Mixtral)
# -------------------------
def hf_generate_text(prompt: str, language: str, max_new_tokens: int = 512, temperature: float = 0.2) -> Tuple[Optional[str], Optional[str]]:
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured (HF_API_KEY)."

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"You are KrishiSakhi, a helpful, knowledgeable agriculture advisor for Karnataka. Respond only in {'Kannada' if language=='kn' else 'English'}."

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt} 
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"HF non-200: {response.status_code} {response.text}")
            return None, f"HF error {response.status_code}: {response.text}"

        data = response.json()
        if not data.get("choices") or not data["choices"][0].get("message"):
            return None, "HF response format invalid."

        text = data["choices"][0]["message"]["content"]
        return text, None

    except Exception as e:
        logger.warning(f"HF exception: {e}")
        return None, str(e)


# -------------------------
# Intent engine (Router)
# -------------------------
def route(query: str, user_id: str, lang: str, session_key: str):
    q = (query or "").lower().strip()
    
    # Custom Router Logic based on keywords
    if any(tok in q for tok in ["soil test", "soil center"]):
        t, v, s = soil_testing_center(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["timeline", "activity log"]):
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
    if any(tok in q for tok in ["pest", "disease", "leaf", "spots", "yellowing"]):
        diag_text, voice, sugg = diagnose_advanced(query, user_crop=None, lang=lang)
        return {"response_text": diag_text, "voice": voice, "suggestions": sugg}
    if "fertilizer" in q or "irrigation" in q:
        # Placeholder for complex context extraction
        return {"response_text": "Need more context (crop/stage).", "voice": False, "suggestions": ["Fertilizer for paddy tillering"]}
    
    # Default -> HF Mixtral advisory
    hf_prompt = f"Provide short actionable crop advice for the user query: {query}"
    text, err = hf_generate_text(hf_prompt, lang, max_new_tokens=256, temperature=0.2)
    
    if text:
        return {"response_text": text, "voice": False, "suggestions": ["Crop stage","Pest check","Soil test"]}
    else:
        # HF missing/failure -> fallback small canned message
        fallback = "AI advisor currently unavailable. I can still help with soil test, fertilizer, irrigation, pest diagnosis. Ask one of those." if lang=="en" else "AI ಸಲಹೆ ಸದ್ಯ ಲಭ್ಯವಿಲ್ಲ. ಮಣ್ಣು ಪರೀಕ್ಷೆ, ಎರೆ, ನೀರಾವರಿ, ಕೀಟನಿರ್ಣಯಕ್ಕೆ ಕೇಳಿ."
        logger.warning("HF fallback used: %s", err)
        return {"response_text": fallback, "voice": False, "suggestions": ["Soil test","Fertilizer","Pest check"]}

# -------------------------
# Endpoint
# -------------------------
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    user_query = (payload.user_query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    lang = "en"
    session_key = payload.session_id or f"{payload.user_id}-{lang}"

    # 1. Determine Language from Firebase
    try:
        pref = firebase_get(f"Users/{payload.user_id}/farmDetails/preferredLanguage")
        if isinstance(pref, str) and pref.lower() == "kn":
            lang = "kn"
        session_key = f"{payload.user_id}-{lang}"
        
    except Exception as e:
        logger.warning("Language lookup error, defaulting to English.")
    
    # 2. Route Query to Handler
    try:
        if user_query == "INITIAL_LOAD":
            # Initial load triggers a welcome message, not a standard route
            welcome_msg = "ನಾನು ಈಗ ನಿಮ್ಮ ಆದ್ಯತೆಯ ಭಾಷೆಯಲ್ಲಿ (ಕನ್ನಡದಲ್ಲಿ) ಉತ್ತರಿಸಲು ಸಿದ್ಧವಾಗಿದ್ದೇನೆ." if lang == 'kn' else "I am ready to assist you in English. How can I help with your farming today?"
            result = {"response_text": welcome_msg, "voice": True, "suggestions": ["Crop stage", "Pest check", "Weather"]}
        else:
            result = route(user_query, payload.user_id, lang, session_key)
            
    except Exception as e:
        logger.exception("Processing error: %s", e)
        fallback_msg = "A critical error occurred while processing your request. Please try again." if lang == "en" else "ಪ್ರಕ್ರಿಯೆಯಲ್ಲಿ ದೋಷ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        raise HTTPException(status_code=500, detail=f"Processing error: {fallback_msg}")

    # 3. Generate TTS Audio
    audio_url = None
    try:
        if result.get("response_text"):
            audio_url = generate_tts_audio(result["response_text"], lang)
    except Exception as e:
        logger.exception("TTS generation failed: %s", e)

    # 4. Return Final Response
    return ChatResponse(
        session_id=session_key,
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
    os.makedirs(TTS_DIR, exist_ok=True)
    
    try:
        initialize_firebase_credentials()
    except Exception as e:
        logger.exception("Firebase init failed: %s", e)
        
    if not HF_API_KEY:
        logger.warning("HF_API_KEY not configured — HF generation disabled.")
    if not GTTS_AVAILABLE:
        logger.warning("gTTS not installed — TTS disabled.")

# -------------------------
# Simple health endpoint
# -------------------------
@app.get("/health")
def health():
    return {"status":"ok", "time": datetime.utcnow().isoformat(), "hf": bool(HF_API_KEY), "gtts": GTTS_AVAILABLE, "firebase_auth": bool(credentials)}
