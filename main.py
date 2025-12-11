# =========================================================
# main.py — Optimized KS Chatbot Backend (FastAPI + Gemini)
# =========================================================

import os
import json
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")

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

app = FastAPI(title="KS Chatbot Backend", version="2.0")


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
    metadata: Optional[Dict[str, Any]]


# =========================================================
# INITIALIZATION HELPERS
# =========================================================
def initialize_gemini():
    global client
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini initialized.")
    except Exception as e:
        print("Failed to initialize Gemini:", e)


def initialize_firebase_credentials():
    global credentials
    if credentials:
        return
    try:
        info = json.loads(SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        print("Firebase service account loaded.")
    except Exception as e:
        print("FATAL: Cannot load Firebase credentials:", e)
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
# FIREBASE HELPERS
# =========================================================
def firebase_get(path: str):
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_DATABASE_URL}/{path}.json"
        response = requests.get(url, params={"access_token": token}, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Firebase GET failed:", e)
        return None


def get_language(user_id: str) -> str:
    lang = firebase_get(f"Users/{user_id}/preferredLanguage")
    if isinstance(lang, str):
        lang = lang.lower()
        return "kn" if lang == "kn" else "en"
    return "en"


def get_user_location(user_id: str):
    data = firebase_get(f"Users/{user_id}/farmDetails")
    if not isinstance(data, dict):
        return None
    district = data.get("district")
    taluk = data.get("taluk")

    if district and taluk:
        return {"district": district, "taluk": taluk}
    return None


# =========================================================
# MODULE: SOIL TESTING CENTER
# =========================================================
def soil_testing_center(user_id, language):
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
        msg = {
            "en": "No soil test center found for your area.",
            "kn": "ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ಮಣ್ಣು ಪರೀಕ್ಷಾ ಕೇಂದ್ರ ಲಭ್ಯವಿಲ್ಲ."
        }
        return msg[language], True, ["Update farm details"]

    for _, info in centers.items():
        if isinstance(info, dict):
            text = f"{info.get('name')}\n{info.get('address')}\nContact: {info.get('contact')}"
            return text, True, ["Directions", "Call center"]

    return ("No data available.", True, [])


# =========================================================
# MODULE: WEATHER
# =========================================================
def weather_advice(language):
    msg = {
        "en": "Check weather forecast. If rain expected, delay irrigation. Mulching helps retain soil moisture.",
        "kn": "ಹವಾಮಾನ ವರದಿಯನ್ನು ಪರಿಶೀಲಿಸಿ. ಮಳೆ ಸಾಧ್ಯವಿದ್ದರೆ ನೀರಾವರಿ ತಡೆಯಿರಿ. ಮಲ್ಚಿಂಗ್ ಮಣ್ಣು ತೇವಾಂಶ ಉಳಿಸುತ್ತದೆ."
    }
    return msg[language], True, ["Irrigation schedule", "Mulching"]


# =========================================================
# MODULE: MARKET PRICE
# =========================================================
PRICE_LIST = {
    "chilli": 50, "paddy": 20, "ragi": 18, "areca": 470,
    "banana": 12, "turmeric": 120, "cotton": 40, "sugarcane": 3
}

def market_price(query, language):
    q = query.lower()

    for crop, price in PRICE_LIST.items():
        if crop in q:
            if language == "kn":
                text = f"{crop.title()} ಸರಾಸರಿ ಬೆಲೆ: ₹{price} / ಕಿ.ಗ್ರಾಂ. ಸ್ಥಳೀಯ APMC ಪರಿಶೀಲಿಸಿ."
            else:
                text = f"Approx price for {crop.title()}: ₹{price}/kg. Verify with local APMC."
            return text, False, ["Sell at APMC", "Quality Check"]

    fallback = {
        "en": "Please specify a crop name (e.g., 'chilli price').",
        "kn": "ದಯವಿಟ್ಟು ಬೆಳೆ ಹೆಸರು ನೀಡಿ (ಉದಾ: 'ಮೆಣಸಿನ ಬೆಲೆ')."
    }
    return fallback[language], False, ["Chilli price", "Areca price"]


# =========================================================
# MODULE: PEST/DISEASE
# =========================================================
def pest_disease(query, language):
    q = query.lower()

    cases = [
        ("curl", 
         "Symptoms indicate leaf curl virus. Remove infected shoots and spray neem oil.",
         "ಎಲೆ ಕರ್ಭಟ ವೈರಸ್ ಲಕ್ಷಣ. ಸೋಂಕಿತ ಕೊಂಬೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ.",
         ["Neem spray", "Remove affected shoots"]),

        ("yellow",
         "Yellow leaves suggest nutrient deficiency or overwatering.",
         "ಎಲೆಗಳು ಹಳದಿಯಾಗುವುದು ಪೋಷಕಾಂಶ ಕೊರತೆ ಅಥವಾ ಅಧಿಕ ನೀರಿನ ಲಕ್ಷಣ.",
         ["Soil test", "Balanced fertilizer"]),

        ("spots",
         "Spots suggest fungal infection. Remove infected leaves and apply fungicide.",
         "ಕಲೆಗಳು ಫಂಗಲ್ ಸೋಂಕಿನ ಲಕ್ಷಣ. ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಫಂಗಿಸೈಡ್ ಬಳಸಿ.",
         ["Biofungicide", "Remove infected leaves"])
    ]

    for key, en, kn, sug in cases:
        if key in q:
            return (kn if language == "kn" else en), True, sug

    fallback = {
        "en": "Provide more symptom details or upload a photo for better diagnosis.",
        "kn": "ಲಕ್ಷಣಗಳ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."
    }
    return fallback[language], True, ["Describe symptoms", "Upload photo"]


# =========================================================
# MODULE: FARM ACTIVITY TIMELINE
# =========================================================
def farm_timeline(user_id, language):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")

    if not logs:
        msg = {
            "en": "No farm activity logs found.",
            "kn": "ಫಾರಂ ಚಟುವಟಿಕೆ ಲಾಗ್ ಕಂಡುಬರಲಿಲ್ಲ."
        }
        return msg[language], False, ["Add activity"]

    result = []
    for crop, entries in logs.items():
        if isinstance(entries, dict):
            latest = max(entries.values(), key=lambda x: x.get("timestamp", 0))
            name = latest.get("cropName", crop)
            act = latest.get("subActivity", "")
            stage = latest.get("stage", "")

            text = (f"{name}: latest activity {act} (stage: {stage})"
                    if language == "en"
                    else f"{name}: ಇತ್ತೀಚಿನ ಚಟುವಟಿಕೆ {act} (ಹಂತ: {stage})")

            result.append(text)

    summary = "\n".join(result)
    prefix = "Farm activity summary:\n" if language == "en" else "ಚಟುವಟಿಕೆ ಸಾರಾಂಶ:\n"
    return prefix + summary, False, ["Next steps", "View full timeline"]


# =========================================================
# MODULE: GEMINI CROP ADVISORY
# =========================================================
def crop_advisory(user_id, query, language, session_key):
    global client, active_chats

    try:
        if session_key not in active_chats:
            prompt = get_prompt(language)
            cfg = types.GenerateContentConfig(system_instruction=prompt)
            chat = client.chats.create(model="gemini-2.5-flash", config=cfg)
            active_chats[session_key] = chat

        chat = active_chats[session_key]
        resp = chat.send_message(query)
        text = resp.text if hasattr(resp, "text") else str(resp)

        return text, False, ["Crop stage", "Pest check", "Soil test"], session_key
    except Exception as e:
        return f"AI error: {e}", False, [], session_key


def get_prompt(language):
    lang = "Kannada" if language == "kn" else "English"
    return f"""
You are KrishiSakhi, an agricultural assistant for Karnataka.
Respond ONLY in {lang}. Give short, actionable advice.
"""


# =========================================================
# INTENT ROUTER
# =========================================================
def route(query, user_id, language, session_key):
    q = query.lower()

    # Priority routing
    if any(word in q for word in ["soil test", "soil center", "soil lab"]):
        text, voice, sug = soil_testing_center(user_id, language)
        return {"response_text": text, "voice": voice, "suggestions": sug}

    if any(w in q for w in ["timeline", "activity log", "farm activity"]):
        text, voice, sug = farm_timeline(user_id, language)
        return {"response_text": text, "voice": voice, "suggestions": sug}

    if any(w in q for w in ["weather", "rain", "forecast"]):
        text, voice, sug = weather_advice(language)
        return {"response_text": text, "voice": voice, "suggestions": sug}

    if any(w in q for w in ["price", "market", "mandi"]):
        text, voice, sug = market_price(query, language)
        return {"response_text": text, "voice": voice, "suggestions": sug}

    if any(w in q for w in ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
        text, voice, sug = pest_disease(query, language)
        return {"response_text": text, "voice": voice, "suggestions": sug}

    # Default → Gemini advisory
    text, voice, sug, sid = crop_advisory(user_id, query, language, session_key)
    return {
        "response_text": text,
        "voice": voice,
        "suggestions": sug,
        "session_id": sid
    }


# =========================================================
# ENDPOINT
# =========================================================
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key missing")
    if not SERVICE_ACCOUNT_KEY:
        raise HTTPException(500, "SERVICE_ACCOUNT_KEY missing")

    user_query = payload.user_query.strip()
    if not user_query:
        raise HTTPException(400, "Query cannot be empty.")

    language = get_language(payload.user_id)
    session_key = payload.session_id or f"{payload.user_id}-{language}"

    result = route(user_query, payload.user_id, language, session_key)

    return ChatResponse(
        session_id=result.get("session_id", session_key),
        response_text=result["response_text"],
        language=language,
        suggestions=result.get("suggestions", []),
        voice=result.get("voice", False),
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    initialize_firebase_credentials()
    initialize_gemini()
