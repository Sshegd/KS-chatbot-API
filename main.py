import os
import requests
import json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest # Alias to avoid conflict
from dotenv import load_dotenv

# Load keys from .env for local testing (Render will use its environment variables)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
SERVICE_ACCOUNT_FILE_PATH = os.getenv("SERVICE_ACCOUNT_FILE_PATH", "./service_account_key.json")

# Define the required scopes for Realtime Database access
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email", 
    "https://www.googleapis.com/auth/firebase.database"
]

# --- Global Cache for Token and Credentials ---
# In a serverless environment like Render, this helps manage the token lifetime.
credentials = None 

# --- FastAPI Setup ---
app = FastAPI(title="KS_chatbot Gemini API", version="1.0.0")

# --- Initialization and Token Management ---
def initialize_firebase_credentials():
    """Initializes Google Credentials using the service account file."""
    global credentials
    if credentials is None:
        try:
            # Load credentials from the file placed in the repository
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE_PATH, 
                scopes=SCOPES
            )
            print("Firebase Credentials Initialized.")
        except Exception as e:
            print(f"FATAL: Could not load Service Account Key. Check path and content: {e}")
            # Do not raise exception here, allow FastAPI to start for health checks
            pass

def get_oauth2_access_token() -> str:
    """Refreshes and returns the valid OAuth2 Access Token."""
    
    global credentials
    if credentials is None:
        initialize_firebase_credentials()
        if credentials is None:
             raise Exception("Firebase credentials are not available.")

    try:
        # Check if the token is expired or needs refreshing
        if not credentials.token or credentials.expired:
            request = GoogleAuthRequest()
            credentials.refresh(request) # Refresh the token
        
        return credentials.token
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Authentication service failed: {e}")

# --- AI Configuration (Runs once at startup) ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash") 
except Exception as e:
    print(f"Error configuring Gemini: {e}")

active_chats: dict[str, genai.Chat] = {} 

# --- Pydantic Data Models (Same as previous, required for API contract) ---
class ChatQuery(BaseModel):
    user_id: str 
    user_query: str
    session_id: str 

class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    language: str 

# --- Firebase Integration Logic ---
def get_language_preference(user_id: str) -> str:
    """Fetches the preferredLanguage from Firebase RTDB using an OAuth2 Access Token."""
    
    try:
        access_token = get_oauth2_access_token() 
    except HTTPException as e:
        print(f"Token generation failed: {e.detail}")
        return 'en'

    # URL structure: /Users/{user_id}/farmDetails/preferredLanguage
    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/farmDetails/preferredLanguage.json"
    
    # Authenticate via the access_token query parameter
    params = {'access_token': access_token} 
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        
        lang_preference = response.json() 
        
        if isinstance(lang_preference, str):
            lang_code = lang_preference.strip().lower()
            return lang_code if lang_code in ['kn', 'en'] else 'en'
        
        return 'en' 
        
    except requests.exceptions.RequestException as e:
        print(f"Firebase REST API Error for {user_id}: {e}")
        return 'en' 

# --- Core Logic: Comprehensive System Prompt (Same as previous) ---
def get_krishi_sakhi_prompt(language: str) -> str:
    # ... (Prompt logic remains the same) ...
    lang_name = "Kannada" if language == 'kn' else "English"
    return f"""
    You are 'KrishiSakhi', a highly knowledgeable and empathetic agricultural expert specializing in Karnataka, India. 
    ... [Detailed prompt logic] ...
    **CRITICAL CONSTRAINT: Always respond ONLY and ENTIRELY in {lang_name}. Do NOT use any other language.**
    """

# --- API Endpoint ---
@app.post("/chat/send", response_model=ChatResponse)
async def chat_with_gemini(query: ChatQuery):
    
    language_code = get_language_preference(query.user_id)
    session_id = f"{query.user_id}-{language_code}" 
    
    # 1. Initialize Chat Session (Ensures correct language prompt)
    if session_id not in active_chats:
        try:
            system_prompt = get_krishi_sakhi_prompt(language_code)
            config = types.GenerateContentConfig(system_instruction=system_prompt)
            
            chat = model.start_chat(config=config)
            active_chats[session_id] = chat
            
            # Send initial welcome message
            if query.user_query == "INITIAL_LOAD":
                welcome_msg = "ನಾನು ಈಗ ನಿಮ್ಮ ಆದ್ಯತೆಯ ಭಾಷೆಯಲ್ಲಿ (ಕನ್ನಡದಲ್ಲಿ) ಉತ್ತರಿಸಲು ಸಿದ್ಧವಾಗಿದ್ದೇನೆ." if language_code == 'kn' else "I am now ready to assist you in your preferred language (English)."
                return ChatResponse(session_id=session_id, response_text=welcome_msg, language=language_code)
            
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"AI/Session Error: {e}")

    chat = active_chats[session_id]

    # 2. Send Message to Gemini
    try:
        response = chat.send_message(query.user_query)
        
        # 3. Return the response
        return ChatResponse(
            session_id=session_id,
            response_text=response.text,
            language=language_code
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Gemini API Error: {e}")

# Ensure credentials are initialized once when the app starts up
@app.on_event("startup")
def startup_event():
    initialize_firebase_credentials()