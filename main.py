# main.py
import os
import json
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# Load environment (local dev)
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")  # JSON string

if FIREBASE_DATABASE_URL and FIREBASE_DATABASE_URL.endswith("/"):
    FIREBASE_DATABASE_URL = FIREBASE_DATABASE_URL.rstrip("/")

# -----------------------------
# Scopes and globals
# -----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]

credentials = None
client = None
active_chats: Dict[str, Any] = {}  # session_key -> chat session object

app = FastAPI(title="KS Chatbot - RuleEngine Backend", version="1.0.0")


# -----------------------------
# Pydantic models
# -----------------------------
class ChatQuery(BaseModel):
    user_id: str
    user_query: str
    session_id: Optional[str] = None  # client may send session_id; if not present server will create/derive


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    language: str
    suggestions: Optional[List[str]] = None
    voice: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None


# -----------------------------
# Helper: initialize Gemini
# -----------------------------
def initialize_gemini_client():
    global client
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client initialized.")
    except Exception as e:
        print("Could not initialize Gemini client:", e)


# -----------------------------
# Helper: initialize Firebase credentials
# -----------------------------
def initialize_firebase_credentials():
    global credentials
    if credentials is not None:
        return

    try:
        if not SERVICE_ACCOUNT_KEY:
            raise Exception("SERVICE_ACCOUNT_KEY env missing")

        info = json.loads(SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        print("Firebase credentials loaded from SERVICE_ACCOUNT_KEY.")
    except Exception as e:
        print("FATAL: Firebase credentials load failed:", e)


# -----------------------------
# Helper: get OAuth2 token for Firebase REST reads
# -----------------------------
def get_oauth2_access_token() -> str:
    global credentials
    if credentials is None:
        initialize_firebase_credentials()
        if credentials is None:
            raise HTTPException(status_code=500, detail="Firebase credentials not available")

    try:
        if not credentials.token or credentials.expired:
            request = GoogleAuthRequest()
            credentials.refresh(request)
        return credentials.token
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed refreshing token: {e}")


# -----------------------------
# Helper: Fetch preferred language (correct path)
# -----------------------------
def get_language_preference(user_id: str) -> str:
    try:
        token = get_oauth2_access_token()
    except HTTPException:
        return "en"

    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/preferredLanguage.json"
    try:
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        val = r.json()
        if isinstance(val, str):
            val = val.lower().strip()
            return val if val in ["en", "kn"] else "en"
    except Exception:
        pass
    return "en"


# -----------------------------
# Helper: Get farmDetails district/taluk
# -----------------------------
def get_user_district_and_taluk(user_id: str) -> Optional[Dict[str, str]]:
    try:
        token = get_oauth2_access_token()
    except HTTPException:
        return None

    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/farmDetails.json"
    try:
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        district = data.get("district")
        taluk = data.get("taluk")
        if district and taluk:
            return {"district": district, "taluk": taluk}
    except Exception:
        pass
    return None


# -----------------------------
# Module: Soil testing center finder
# -----------------------------
def find_soil_testing_center(user_id: str) -> Optional[Dict[str, str]]:
    loc = get_user_district_and_taluk(user_id)
    if not loc:
        return None

    district = loc["district"]
    taluk = loc["taluk"]

    # Build path under SoilTestingCenters/Karnataka/{district}/{taluk}
    try:
        token = get_oauth2_access_token()
    except HTTPException:
        return None

    url = f"{FIREBASE_DATABASE_URL}/SoilTestingCenters/Karnataka/{district}/{taluk}.json"
    try:
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        centers = r.json()
        if not centers:
            return None
        # return first center's details
        for key, info in centers.items():
            if isinstance(info, dict):
                return {
                    "name": info.get("name", ""),
                    "address": info.get("address", ""),
                    "contact": info.get("contact", "")
                }
    except Exception:
        pass
    return None


# -----------------------------
# Module: Simple weather advisory (rule-based placeholder)
# -----------------------------
def weather_advice(user_id: str, language: str) -> Dict[str, Any]:
    # For a production system, integrate an actual weather API (e.g., OpenWeatherMap) using user's farm coordinates/district.
    # Here we use a simple rule-based heuristic and produce voice-friendly short advice.
    advice_text = {
        "en": "Check local forecast. If rain expected within 24 hours, delay irrigation. Use mulching to conserve soil moisture.",
        "kn": "ಸ್ಥಳೀಯ ಹವಾಮಾನವನ್ನು ಪರಿಶೀಲಿಸಿ. 24 ಗಂಟೆಗಳಲ್ಲಿ ಮಳೆ ಸಾದ್ಯವಾದರೆ ನೀರಾವರಿವನ್ನವ ತಡೆಯಿರಿ. ಮಲ್ಚ್ ಬಳಸಿ ಮಣ್ಣು ஈರವತೆಯನ್ನು ಉಳಿಸಿ."
    }
    return {"advice": advice_text.get(language, advice_text["en"]), "voice": True, "suggestions": ["Irrigation schedule", "Soil moisture check", "Mulching"]}


# -----------------------------
# Module: Market price guidance (mock rule-based)
# -----------------------------
def market_price_advice(query: str, language: str) -> Dict[str, Any]:
    # Very simple mock. In production integrate mandi/APMC price feeds.
    # Look for crop name heuristically in query.
    crop_lookup = ["chilli", "paddy", "ragi", "areca", "areca nut", "banana", "turmeric", "cotton", "sugarcane"]
    q_low = query.lower()
    crop_found = None
    for c in crop_lookup:
        if c in q_low:
            crop_found = c
            break

    if crop_found:
        # mock price values (INR/kg)
        mock_prices = {
            "chilli": 50,
            "paddy": 20,
            "ragi": 18,
            "areca": 470,
            "banana": 12,
            "turmeric": 120,
            "cotton": 40,
            "sugarcane": 3
        }
        price = mock_prices.get(crop_found, None)
        if language == "kn":
            text = f"{crop_found.title()} ನಿಲುವಿಗೆ ಸರಾಸರಿ ಬೆಲೆ: ₹{price} ಪ್ರತಿ ಕಿ.ಗ್ರಾ. - ಸ್ಥಳೀಯ APMC ವೀಕ್ಷಿಸಿ."
        else:
            text = f"Approx price for {crop_found.title()}: ₹{price} per kg. Verify with local APMC before selling."
        return {"advice": text, "voice": False, "suggestions": ["Sell at APMC", "Store & check quality", "Contact trader"]}
    else:
        fallback = {
            "en": "Please specify the crop (e.g., 'chilli price'). For exact mandi rates check your local APMC.",
            "kn": "ದಯವಿಟ್ಟು ಮೆಲುಕು ನ ಬೆವರು (ಉದಾ: 'ಮೆಣಸಿನ ಬೆಲೆ') ತಿಳಿಸಿ. ನಿಖರದಿಟ್ಟಿಗೆ ಸ್ಥಳೀಯ APMC ನೋಡಿರಿ."
        }
        return {"advice": fallback.get(language, fallback["en"]), "voice": False, "suggestions": ["Chilli price", "Areca price"]}


# -----------------------------
# Module: Pest/Disease analysis (small knowledge base)
# -----------------------------
def pest_disease_analysis(query: str, language: str) -> Dict[str, Any]:
    q = query.lower()
    # simple symptom mapping
    if "leaf curl" in q or "leaves curl" in q or "curl" in q:
        advice_en = ("Symptoms indicate leaf curl virus or sucking pests. "
                     "Initial action: remove severely affected shoots and apply neem oil spray. "
                     "If condition worsens, contact Krishi Adhikari.")
        advice_kn = ("ಎಲೆ ಕರ್ಭಟವು ವೈರಸ್ ಅಥವಾ ಸಗಟುದ ಹಾಳುಕಾರಕಗಳ ಸೂಚನೆ. ಪ್ರಾಥಮಿಕ ಕ್ರಮ: ಗಂಭೀರವಾಗಿ ಪ್ರಭಾವಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ. ಸ್ಥಿತಿ ಕೆಟ್ಟದಾದರೆ ಕೃಷಿ ಅಧಿಕಾರಿಗೆ ಸಂಪರ್ಕಿಸಿ.")
        return {"advice": advice_kn if language == "kn" else advice_en,
                "voice": True,
                "suggestions": ["Neem oil spray", "Remove affected parts", "Contact Krishi Adhikari"]}
    if "yellow" in q or "yellowing" in q:
        advice_en = ("Yellowing leaves may indicate nutrient deficiency (N/P/K) or overwatering. "
                     "Check soil moisture and consider soil test. Apply balanced fertilizer as per soil report.")
        advice_kn = ("ಎಲೆಗಳನ್ನು ಪಿಚ್ಚಾಗಿಸುವುದು ಉತ್ತರದರಿದಾಗಿದೆ ಅಥವಾ ಹೆಚ್ಚುವರಿ ನೀರು ಕಾರಣವಾಗಿರಬಹುದು. ಮಣ್ಣಿನ ஈರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿ.")
        return {"advice": advice_kn if language == "kn" else advice_en,
                "voice": True,
                "suggestions": ["Soil test", "Nitrogen application", "Improve drainage"]}
    if "spots" in q or "blight" in q or "fungus" in q:
        advice_en = ("Spots/blight suggest fungal infection. Initial action: remove infected leaves, improve air circulation, apply recommended fungicide or biofungicide (e.g., neem-based).")
        advice_kn = ("ಕಲೆ/ಬಿಟ್ಟಲುಗಳು ಸೂಕ್ಷ್ಮಾಂಶದ ಸೋಂಕುಗಳ ಸೂಚನೆ. ಪ್ರಾಥಮಿಕ ಕ್ರಮ: ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಸೂಕ್ತ ಫಂಗಿಸೈಡ್ ಬಳಸಿ.")
        return {"advice": advice_kn if language == "kn' else advice_en,
                "voice": True,
                "suggestions": ["Remove infected leaves", "Biofungicide (Neem)", "Consult Krishi Adhikari"]}
    # fallback
    fallback = {
        "en": "Provide more details about symptoms (leaf color, spots, pests seen). If possible, send a photo to local KVK.",
        "kn": "ಲಕ್ಷಣಗಳ ಮಾಹಿತಿ (ಎಲೆಗಳ ಬಣ್ಣ, ಕಲೆಗಳು, ಕಂಡ ಹಾಳುಕಾರಕಗಳು) ನೀಡಿ. ಚಿತ್ರೀಕರಣವನ್ನು ಕೇವಲ KVKಗೆ ಕಳುಹಿಸುವುದು ಉತ್ತಮ."
    }
    return {"advice": fallback.get(language, fallback["en"]), "voice": True, "suggestions": ["Upload photo", "Describe symptoms"]}


# -----------------------------
# Module: Farm activity timeline advisory
# -----------------------------
def farm_activity_timeline_advice(user_id: str, language: str) -> Dict[str, Any]:
    # Fetch farmActivityLogs and analyze latest entries
    try:
        token = get_oauth2_access_token()
    except HTTPException:
        return {"advice": "Could not fetch farm logs.", "voice": False, "suggestions": []}

    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/farmActivityLogs.json"
    try:
        r = requests.get(url, params={"access_token": token}, timeout=10)
        r.raise_for_status()
        logs = r.json()
        # logs structure: cropKey -> {activityId -> {...}}
        latest_summaries = []
        if not logs:
            return {"advice": ("No activity logs found. Add farm activity entries to get timeline advice."
                               if language == "en" else "ಕೃಷಿ ಚಟುವಟಿಕೆ ಲಾಗ್ ಕಂಡುಬರಲಿಲ್ಲ. ಸಲಹೆಗೆ ಚಟುವಟಿಕೆ ಸೇರಿಸಿ."),
                    "voice": False, "suggestions": ["Add activity", "Soil test"]}

        for cropKey, cropLogs in logs.items():
            # iterate activity entries and pick the most recent by timestamp
            latest_entry = None
            latest_ts = -1
            for act_id, data in (cropLogs.items() if isinstance(cropLogs, dict) else []):
                ts = data.get("timestamp", 0)
                if ts and ts > latest_ts:
                    latest_ts = ts
                    latest_entry = data
            if latest_entry:
                crop_name = latest_entry.get("cropName", cropKey)
                stage = latest_entry.get("stage", "unknown")
                activity = latest_entry.get("subActivity", "")
                date = latest_entry.get("activityDate", None)
                if language == "kn":
                    summary = f"{crop_name}: ಅತಿ تازಾ ಚಟುವಟಿಕೆ {activity} (ಸೈಟೆ: {stage})"
                else:
                    summary = f"{crop_name}: latest activity {activity} (stage: {stage})"
                latest_summaries.append(summary)

        if not latest_summaries:
            return {"advice": ("No recent activities found." if language == "en" else "ಯಾವುದೇ ಇತ್ತೀಚಿನ ಚಟುವಟಿಕೆಗಳು ಕಾಣಿಬರಲಿಲ್ಲ."),
                    "voice": False, "suggestions": ["Record activity", "Soil test"]}

        big_text = "\n".join(latest_summaries)
        advice_prefix = "Farm timeline summary:\n" if language == "en" else "ತೋಟದ ಚಟುವಟಿಕೆ ಸಾರಾಂಶ:\n"
        return {"advice": advice_prefix + big_text, "voice": False, "suggestions": ["View full timeline", "Next action suggestions"]}
    except Exception:
        return {"advice": ("Failed reading activity logs." if language == "en" else "ಚಟುವಟಿಕೆ ಲಾಗ್ ಓದಲು ವಿಫಲ."), "voice": False, "suggestions": []}


# -----------------------------
# Module: Crop advisory via Gemini (fallback and detailed advice)
# -----------------------------
def crop_advisory_via_gemini(user_id: str, user_query: str, language: str, session_key: str) -> Dict[str, Any]:
    # Ensure chat session exists for session_key
    global client, active_chats
    try:
        if session_key not in active_chats:
            prompt = get_krishi_sakhi_prompt(language)
            config = types.GenerateContentConfig(system_instruction=prompt)
            # NOTE: use client.chats.create to create a chat session
            chat = client.chats.create(model="gemini-2.5-flash", config=config)
            active_chats[session_key] = chat
        else:
            chat = active_chats[session_key]

        # send user's query to chat session and get response
        resp = chat.send_message(user_query)
        text = resp.text if hasattr(resp, "text") else str(resp)
        return {"advice": text, "voice": False, "suggestions": ["Follow-up: crop stage", "Soil test", "Pest check"], "session_id": session_key}
    except Exception as e:
        return {"advice": f"AI service error: {e}", "voice": False, "suggestions": []}


# -----------------------------
# System prompt helper
# -----------------------------
def get_krishi_sakhi_prompt(language: str) -> str:
    lang_name = "Kannada" if language == "kn" else "English"
    return f"""
You are KrishiSakhi, a knowledgeable and empathetic agricultural assistant for Karnataka.
Respond ONLY in {lang_name}. Be concise and actionable. Provide:
- Crop-specific guidance (Paddy, Ragi, Sugarcane, Turmeric, Cotton)
- Pest & disease identification: Identification, Non-chemical IPM actions, When to consult Krishi Adhikari
- When mentioning schemes or prices, always add a disclaimer to verify with local authorities.
If user asks for soil center or market price, return a short answer and quick suggestions.
"""


# -----------------------------
# Intent router (Option B rules)
# -----------------------------
def route_query(query: str, user_id: str, language: str, session_key: str) -> Dict[str, Any]:
    q = query.lower()

    # Priority rules (soil center and farm timeline use Firebase)
    # Soil testing center keywords
    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center", "soil lab", "soil centre", "soil centre"]):
        center = find_soil_testing_center(user_id)
        if center:
            text = f"{center.get('name', '')}\n{center.get('address', '')}\nContact: {center.get('contact', '')}"
            return {"response_text": text, "suggestions": ["Directions", "Call center"], "voice": True}
        else:
            msg = {"en": "No soil testing center found for your taluk/district. Please update farmDetails with district and taluk.",
                   "kn": "ನಿಮ್ಮ ತಾಲೂಕು/ಜಿಲ್ಲೆಗಾಗಿ ಮಣ್ಣು ಪರೀಕ್ಷೆ ಕೇಂದ್ರ ಕಂಡುಬರಲಿಲ್ಲ. ದಯವಿಟ್ಟು farmDetails ನಲ್ಲಿ ಜಿಲ್ಲೆ ಮತ್ತು ತಾಲೂಕು ಸರಿ ಮಾಡಿ."}
            return {"response_text": msg.get(language, msg["en"]), "suggestions": ["Update farm details"], "voice": True}

    # Farm timeline
    if any(tok in q for tok in ["timeline", "activity log", "activity", "farm activity", "history", "timeline of my farm"]):
        res = farm_activity_timeline_advice(user_id, language)
        return {"response_text": res["advice"], "suggestions": res.get("suggestions", []), "voice": res.get("voice", False)}

    # Weather
    if "weather" in q or "rain" in q or "forecast" in q:
        res = weather_advice(user_id, language)
        return {"response_text": res["advice"], "suggestions": res.get("suggestions", []), "voice": res.get("voice", False)}

    # Market price
    if "price" in q or "market" in q or "mandi" in q:
        res = market_price_advice(query, language)
        return {"response_text": res["advice"], "suggestions": res.get("suggestions", []), "voice": res.get("voice", False)}

    # Pest / disease
    pest_keywords = ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]
    if any(tok in q for tok in pest_keywords):
        res = pest_disease_analysis(query, language)
        return {"response_text": res["advice"], "suggestions": res.get("suggestions", []), "voice": res.get("voice", True)}

    # Otherwise, treat as crop advisory and send to Gemini
    res = crop_advisory_via_gemini(user_id, query, language, session_key)
    return {"response_text": res.get("advice", "Sorry, I couldn't answer that."), "suggestions": res.get("suggestions", []), "voice": res.get("voice", False), "session_id": res.get("session_id", session_key)}


# -----------------------------
# Main chat endpoint (rule-based)
# -----------------------------
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(query: ChatQuery):
    # sanity checks
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    if not FIREBASE_DATABASE_URL:
        raise HTTPException(status_code=500, detail="FIREBASE_DATABASE_URL not configured.")
    if not SERVICE_ACCOUNT_KEY:
        raise HTTPException(status_code=500, detail="SERVICE_ACCOUNT_KEY not configured.")

    user_id = query.user_id
    user_query = (query.user_query or "").strip()
    if not user_id or not user_query:
        raise HTTPException(status_code=400, detail="user_id and user_query are required.")

    # determine language (from Firebase user preferredLanguage)
    language = get_language_preference(user_id)

    # session key: user-language stable session
    session_key = query.session_id if query.session_id else f"{user_id}-{language}"

    # route by rules
    try:
        routed = route_query(user_query, user_id, language, session_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {e}")

    response_text = routed.get("response_text", "Sorry, could not process.")
    suggestions = routed.get("suggestions", [])
    voice_flag = bool(routed.get("voice", False))
    session_id_ret = routed.get("session_id", session_key)

    # build metadata (optional) — e.g., include source (rule/module name)
    metadata = {
        "source": "rule_engine",
        "timestamp": datetime.utcnow().isoformat()
    }

    return ChatResponse(
        session_id=session_id_ret,
        response_text=response_text,
        language=language,
        suggestions=suggestions,
        voice=voice_flag,
        metadata=metadata
    )


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    initialize_firebase_credentials()
    initialize_gemini_client()
