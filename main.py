import os
import requests
import json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv

# ----------------------------------------
# LOAD ENV (local only)
# ----------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")  # JSON string

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]

credentials = None
client = None
active_chats = {}

app = FastAPI(title="KS Chatbot API", version="3.0.0")


# ----------------------------------------
# MODELS
# ----------------------------------------
class ChatQuery(BaseModel):
    user_id: str
    user_query: str
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    language: str


# ----------------------------------------
# INITIALIZE GEMINI
# ----------------------------------------
def initialize_gemini_client():
    global client
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client Initialized.")
    except Exception as e:
        print(f"FATAL: Gemini Init Failed → {e}")


# ----------------------------------------
# INITIALIZE FIREBASE
# ----------------------------------------
def initialize_firebase_credentials():
    global credentials

    if credentials:
        return

    try:
        if not SERVICE_ACCOUNT_KEY:
            raise Exception("SERVICE_ACCOUNT_KEY missing in Render environment.")

        info = json.loads(SERVICE_ACCOUNT_KEY)

        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )

        print("Firebase Credentials Loaded.")

    except Exception as e:
        print(f"FATAL Firebase Cred Error: {e}")


# ----------------------------------------
# OAUTH TOKEN
# ----------------------------------------
def get_oauth2_access_token():
    global credentials

    if credentials is None:
        initialize_firebase_credentials()

    try:
        if not credentials.token or credentials.expired:
            credentials.refresh(GoogleAuthRequest())

        return credentials.token

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token Error: {e}")


# ----------------------------------------
# GET LANGUAGE PREFERENCE
# ----------------------------------------
def get_language_preference(user_id: str) -> str:
    try:
        token = get_oauth2_access_token()
    except:
        return "en"

    # FIXED path: correct location for preferredLanguage
    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/preferredLanguage.json"

    try:
        res = requests.get(url, params={"access_token": token})
        res.raise_for_status()

        lang = res.json()
        if isinstance(lang, str):
            lang = lang.lower()
            return lang if lang in ["en", "kn"] else "en"

        return "en"

    except:
        return "en"


# ----------------------------------------
# SYSTEM PROMPT
# ----------------------------------------
def get_prompt(language: str):
    lang = "Kannada" if language == "kn" else "English"

    return f"""
You are KrishiSakhi, a Karnataka agriculture assistant.

Rules:
1. ALWAYS reply ONLY in {lang}.
2. Provide guidance on: Paddy, Ragi, Sugarcane, Cotton, Turmeric.
3. Follow IPM for pest management.
4. If schemes are mentioned → add disclaimer to verify with local officers.
5. Keep responses simple, helpful, farmer-friendly.
"""


# ----------------------------------------
# CHAT ENDPOINT
# ----------------------------------------
@app.post("/chat/send", response_model=ChatResponse)
async def chat_with_gemini(query: ChatQuery):

    if client is None:
        raise HTTPException(503, "Gemini client not initialized.")

    language = get_language_preference(query.user_id)
    session_key = f"{query.user_id}-{language}"

    # CREATE NEW SESSION
    if session_key not in active_chats:

        try:
            prompt = get_prompt(language)

            config = types.GenerateContentConfig(system_instruction=prompt)

            # FIXED: correct way to start chat session
            chat = client.chats.create(model="gemini-2.5-flash", config=config)

            active_chats[session_key] = chat

            if query.user_query == "INITIAL_LOAD":
                welcome = "ನಾನು ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಲಿದ್ದೇನೆ." if language == "kn" else "I will answer in English."
                return ChatResponse(session_id=session_key, response_text=welcome, language=language)

        except Exception as e:
            raise HTTPException(500, f"Session Error: {e}")

    chat = active_chats[session_key]

    # SEND MESSAGE
    try:
        response = chat.send_message(query.user_query)

        return ChatResponse(
            session_id=session_key,
            response_text=response.text,
            language=language
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini Error: {e}")


# ----------------------------------------
# STARTUP
# ----------------------------------------
@app.on_event("startup")
def startup_event():
    initialize_firebase_credentials()
    initialize_gemini_client()
