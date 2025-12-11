import os
import requests
import json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
# --- CRITICAL CHANGE: Import the Client directly ---
from google import genai 
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest 
from dotenv import load_dotenv

# Load keys from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
SERVICE_ACCOUNT_FILE_PATH = os.getenv("SERVICE_ACCOUNT_FILE_PATH", "./service_account_key.json")

# Define the required scopes
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email", 
    "https://www.googleapis.com/auth/firebase.database"
]

# --- Global Client and Cache ---
credentials = None 
# Initialize the Gemini Client Object
client = None 

# --- FastAPI Setup ---
app = FastAPI(title="KS_chatbot Gemini API", version="1.0.0")

# --- Initialization Functions (Simplified and Combined) ---

def initialize_gemini_client():
    """Initializes the Gemini Client."""
    global client
    try:
        # Use the standard Client() method with the API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini Client Initialized.")
    except Exception as e:
        print(f"FATAL: Could not initialize Gemini Client: {e}")
        pass

def initialize_firebase_credentials():
    """Initializes Google Credentials for Firebase."""
    global credentials
    if credentials is None:
        try:
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE_PATH, 
                scopes=SCOPES
            )
            print("Firebase Credentials Initialized.")
        except Exception as e:
            print(f"FATAL: Could not load Service Account Key. Check path and content: {e}")
            pass

def get_oauth2_access_token() -> str:
    # ... (Token generation logic remains the same, using the global 'credentials' object) ...
    global credentials
    if credentials is None:
        initialize_firebase_credentials()
        if credentials is None:
             raise Exception("Firebase credentials are not available.")

    try:
        # Check if the token is expired or needs refreshing
        if not credentials.token or credentials.expired:
            request = GoogleAuthRequest()
            credentials.refresh(request)
        
        return credentials.token
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Authentication service failed: {e}")


# --- GLOBAL CHAT CACHE (Uses the correct type hint, though dynamic typing is easier) ---
# NOTE: The type hint can be tricky. Using the object itself is safer:
active_chats = {} 


# --- Firebase Integration Logic (Unchanged) ---
def get_language_preference(user_id: str) -> str:
    # ... (Logic remains the same, using get_oauth2_access_token()) ...
    try:
        access_token = get_oauth2_access_token() 
    except HTTPException as e:
        print(f"Token generation failed: {e.detail}")
        return 'en'

    url = f"{FIREBASE_DATABASE_URL}/Users/{user_id}/farmDetails/preferredLanguage.json"
    params = {'access_token': access_token} 
    
    # ... (requests.get logic remains the same) ...
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


# --- Core Logic: Comprehensive System Prompt (Unchanged) ---
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
    
    # CRITICAL CHECK: Ensure client is initialized
    if client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail="AI service client is not initialized.")

    language_code = get_language_preference(query.user_id)
    session_id = f"{query.user_id}-{language_code}" 
    
    # 1. Initialize Chat Session 
    if session_id not in active_chats:
        try:
            system_prompt = get_krishi_sakhi_prompt(language_code)
            config = types.GenerateContentConfig(system_instruction=system_prompt)
            
            # --- CRITICAL CHANGE: Access model and chat via the client object ---
            chat = client.models.generate_content_stream(model="gemini-2.5-flash", config=config)
            # chat = client.models.get("gemini-2.5-flash").start_chat(config=config)
            
            # For this conversational model, using generate_content_stream might be too complex 
            # for basic conversation. Let's use the safer start_chat if available.
            
            # Use the safer start_chat method if the SDK supports it for the client:
            chat = client.chats.create(model="gemini-2.5-flash", config=config) # Corrected path for chat object
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

# Ensure credentials and client are initialized once when the app starts up
@app.on_event("startup")
def startup_event():
    initialize_firebase_credentials()
    initialize_gemini_client()
