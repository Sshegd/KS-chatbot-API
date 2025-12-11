import os
import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 1. Startup Configuration
# --------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
SERVICE_ACCOUNT_FILE_PATH = os.getenv("SERVICE_ACCOUNT_FILE_PATH", "./service_account_key.json")

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]

credentials = None
client = None
active_chats = {}

app = FastAPI(title="KS_chatbot Gemini API", version="1.0.0")

# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------

class ChatQuery(BaseModel):
    user_id: str
    user_query: str
    session_id: str

class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    language: str

# --------------------------------------------------------------------------
# 3. Initialization Functions
# --------------------------------------------------------------------------

def initialize_gemini_client():
    global client
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client Initialized.")
    except Exception as e:
        print(f"FATAL: Could not initialize Gemini Client: {e}")

def initialize_firebase_credentials():
    global credentials
    if credentials is not None:
        return

    try:
        # Get the JSON string from Render environment variable
        service_account_json = os.getenv("SERVICE_ACCOUNT_KEY")

        if not service_account_json:
            raise Exception("SERVICE_ACCOUNT_KEY environment variable is missing.")

        # Parse the JSON string
        service_account_info = json.loads(service_account_json)

        # Create credentials from JSON dict
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )

        print("Firebase Credentials Initialized from environment variable.")

    except Exception as e:
        print(f"FATAL: Could not load Service Account Key from environment variable: {e}")

def get_oauth2_access_token() -> str:
    global credentials

    if credentials is None:
        initialize_firebase_credentials()
        if credentials is None:
            raise Exception("Firebase credentials not available.")

    try:
        if not credentials.token or credentials.expired:
            request = GoogleAuthRequest()
            credentials.refresh(request)

        return credentials.token

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {e}"
        )

# --------------------------------------------------------------------------
# Firebase Fetch
# --------------------------------------------------------------------------

def get_language_preference(user_id: str) -> str:
    try:
        token = get_oauth2_access_token()
    except:
        return "en"

    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/farmDetails/preferredLanguage.json"
    params = {"access_token": token}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        lang = response.json()
        if isinstance(lang, str):
            lang = lang.lower().strip()
            return lang if lang in ["en", "kn"] else "en"

        return "en"

    except:
        return "en"

# --------------------------------------------------------------------------
# System Prompt
# --------------------------------------------------------------------------

def get_krishi_sakhi_prompt(language: str) -> str:
    lang_name = "Kannada" if language == "kn" else "English"
    return f"""
    You are 'KrishiSakhi', an agricultural expert for Karnataka.

    Always respond ONLY in {lang_name}.

    Crop focus: Paddy, Ragi, Sugarcane, Turmeric, Cotton.

    If a pest/disease is mentioned, always include:
    A) Identification
    B) IPM/Non-chemical first steps
    C) When to consult Krishi Adhikari

    For schemes: Add disclaimer to verify locally.

    For off-topic questions:
    Reply in Kannada: "ನಾನು ಕೃಷಿ ಸಂಬಂಧಿತ ವಿಷಯಗಳಿಗೆ ಮಾತ್ರ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ."
    """

# --------------------------------------------------------------------------
# 4. Chat Endpoint
# --------------------------------------------------------------------------

@app.post("/chat/send", response_model=ChatResponse)
async def chat_with_gemini(query: ChatQuery):

    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service not initialized."
        )

    language_code = get_language_preference(query.user_id)
    session_id = f"{query.user_id}-{language_code}"

    # ----------------------------------------
    # CREATE NEW CHAT SESSION
    # ----------------------------------------
    if session_id not in active_chats:
        try:
            system_prompt = get_krishi_sakhi_prompt(language_code)
            config = types.GenerateContentConfig(system_instruction=system_prompt)

            # *** FIXED LINE ***
            chat = client.chat(model="gemini-2.5-flash", config=config)

            active_chats[session_id] = chat

            if query.user_query == "INITIAL_LOAD":
                welcome = (
                    "ನಾನು ಈಗ ನಿಮ್ಮ ಆದ್ಯತೆಯ ಭಾಷೆಯಲ್ಲಿ (ಕನ್ನಡದಲ್ಲಿ) ಉತ್ತರಿಸಲು ಸಿದ್ಧ."
                    if language_code == "kn"
                    else "I am ready to respond in English."
                )
                return ChatResponse(session_id=session_id, response_text=welcome, language=language_code)

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Session Creation Error: {e}"
            )

    chat = active_chats[session_id]

    # ----------------------------------------
    # SEND MESSAGE
    # ----------------------------------------
    try:
        reply = chat.send_message(query.user_query)
        return ChatResponse(
            session_id=session_id,
            response_text=reply.text,
            language=language_code
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API Error: {e}"
        )

# --------------------------------------------------------------------------
# 5. Startup
# --------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    initialize_firebase_credentials()
    initialize_gemini_client()

