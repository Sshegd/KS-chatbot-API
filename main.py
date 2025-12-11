#!/usr/bin/env python3
# main.py - KS Chatbot Backend (FastAPI) — HuggingFace Mixtral (router) integration + features
# Single-file final merged version requested by user (includes many modules)

import os
import json
import requests
import re
import difflib
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env
load_dotenv()

# -----------------------------
# Config / Environment
# -----------------------------
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "").rstrip("/")
SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

if not FIREBASE_DATABASE_URL:
    raise Exception("FIREBASE_DATABASE_URL missing - set environment variable")

# Ensure tts_audio directory exists before StaticFiles mount
TTS_DIR = os.path.join(os.path.dirname(__file__), "tts_audio")
os.makedirs(TTS_DIR, exist_ok=True)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ks-backend")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="KS Chatbot Backend (HF)", version="4.0")

# mount static dir (TTS)
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")

# -----------------------------
# Simple models
# -----------------------------
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
    metadata: Optional[Dict[str, Any]] = None

# -----------------------------
# Globals
# -----------------------------
# firebase credentials placeholder - same pattern as before (we only need token)
credentials = None
active_chats: Dict[str, Any] = {}  # session -> context (here: last prompt memory if desired)

# -----------------------------
# Utility functions
# -----------------------------
def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str):
    return text.split()

# -----------------------------
# Firebase helpers (token via service account)
# -----------------------------
# Note: if SERVICE_ACCOUNT_KEY is a JSON string, we load it; otherwise, fallback logic could be added.
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/firebase.database"
]

def initialize_firebase_credentials():
    global credentials
    if credentials:
        return
    if not SERVICE_ACCOUNT_KEY:
        logger.warning("SERVICE_ACCOUNT_KEY not set; firebase_get will likely fail for auth-protected DBs.")
        return
    try:
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
            raise HTTPException(status_code=500, detail="Firebase credentials not available")
    try:
        if not credentials.valid or credentials.expired:
            credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception as e:
        logger.exception("Token refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {e}")

def firebase_get(path: str):
    """
    Performs a GET request to Firebase Realtime Database (expects SERVICE_ACCOUNT_KEY with DB permissions).
    Returns parsed JSON or None on failure.
    """
    try:
        token = get_firebase_token()
        url = f"{FIREBASE_DATABASE_URL}/{path}.json"
        r = requests.get(url, params={"access_token": token}, timeout=12)
        r.raise_for_status()
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Firebase GET error for path %s: %s", path, e)
        return None

# Convenience wrappers that the modules expect
def get_user_farm_details(user_id: str) -> Dict[str, Any]:
    data = firebase_get(f"Users/{user_id}/farmDetails")
    return data if isinstance(data, dict) else {}

def get_language(user_id: str) -> str:
    lang = firebase_get(f"Users/{user_id}/preferredLanguage")
    if isinstance(lang, str) and lang.lower() == "kn":
        return "kn"
    return "en"

def get_user_location(user_id: str):
    farm = get_user_farm_details(user_id)
    if not farm:
        return None
    return {
        "district": farm.get("district"),
        "taluk": farm.get("taluk")
    }

# -----------------------------
# TTS generation
# -----------------------------
def generate_tts_audio(text: str, lang: str):
    """
    Tries to generate an MP3 TTS using gTTS (google). If package not available, fails gracefully.
    Returns path relative to /tts (e.g., "/tts/tts_uuid.mp3") or None.
    """
    if not text:
        return None
    try:
        from gtts import gTTS
    except Exception:
        logger.warning("gtts not installed; skipping audio generation.")
        return None

    try:
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(TTS_DIR, filename)
        tts = gTTS(text=text, lang="kn" if lang == "kn" else "en")
        tts.save(filepath)
        logger.info("Saved TTS file: %s", filepath)
        return f"/tts/{filename}"
    except Exception as e:
        logger.exception("TTS error: %s", e)
        return None

# -----------------------------
# Hugging Face generate (router) wrapper
# -----------------------------
def hf_generate(prompt: str, model: str = HUGGINGFACE_MODEL, max_new_tokens: int = 512, temperature: float = 0.2) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls Hugging Face Router generate endpoint:
      POST https://router.huggingface.co/hf-inference/models/{model}/v1/generate
    Returns (generated_text or None, error_message or None)
    """
    if not HUGGINGFACE_API_KEY:
        return None, "Hugging Face API key not set (HUGGINGFACE_API_KEY)."

    url = f"https://router.huggingface.co/hf-inference/models/{model}/v1/generate"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.95,
        },
        # add options to get plain text output
        "options": {"use_cache": False, "wait_for_model": True}
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        logger.exception("HF request failed: %s", e)
        return None, f"Hugging Face request failed: {e}"

    # handle obvious errors
    if r.status_code == 410:
        return None, "Hugging Face inference endpoint deprecated for this model (410). Use router.huggingface.co with supported model."
    if r.status_code == 401:
        return None, "Hugging Face authentication failed (401). Check API key."
    if r.status_code == 429:
        return None, f"Hugging Face rate limit / quota exceeded: {r.text}"
    if r.status_code >= 400:
        logger.warning("HF non-200: %s %s", r.status_code, r.text)
        try:
            err = r.json()
        except Exception:
            err = r.text
        return None, f"Hugging Face error {r.status_code}: {err}"

    # parse result - router generate returns a JSON with 'generated_text' or 'outputs'
    try:
        j = r.json()
        # New router format: {"generated_text": "..."} or sometimes {"results":[{"text":"..."}]}
        if isinstance(j, dict):
            if "generated_text" in j:
                return j["generated_text"], None
            if "results" in j and isinstance(j["results"], list) and len(j["results"]) > 0:
                # often each result has "text"
                first = j["results"][0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"], None
                # fallback: join tokens / content
            # some models return {"outputs":[{"generated_text":"..."}]}
            if "outputs" in j and isinstance(j["outputs"], list) and len(j["outputs"]) > 0 and isinstance(j["outputs"][0], dict):
                out = j["outputs"][0]
                if "generated_text" in out:
                    return out["generated_text"], None
                if "text" in out:
                    return out["text"], None
        # fallback - try to interpret as list with text segments
        if isinstance(j, list) and len(j) > 0:
            first = j[0]
            if isinstance(first, dict):
                for k in ("generated_text", "text"):
                    if k in first:
                        return first[k], None
        # final fallback - try to take raw text content
        text = r.text
        return text, None
    except Exception as e:
        logger.exception("Failed to parse HF response: %s", e)
        return None, f"Failed to parse HF response: {e}"

# -----------------------------
# Stage recommendations, fertilizer base, pesticide DB, price list, etc.
# (For brevity here we include essential data — keep and expand as needed)
# -----------------------------
PRICE_LIST = {
    "paddy": 20, "ragi": 18, "maize": 22, "jowar": 18, "tur": 45, "moong": 60,
    "urad": 50, "groundnut": 70, "sunflower": 40, "sesame": 120, "sugarcane": 3,
    "cotton": 40, "arecanut": 470, "banana": 12, "turmeric": 120, "banana": 12,
    "coconut": 25, "cotton": 40, "tomato": 18, "brinjal": 22, "onion": 25,
    "potato": 12, "carrot": 16, "capsicum": 30, "ginger": 80, "coffee": 120,
    "mango": 30, "banana": 12, "papaya": 18, "grapes": 45, "sapota": 35
}

STAGE_RECOMMENDATIONS = {
    "paddy": {"nursery": {"en":"Maintain 2–3 cm water level; protect seedlings.","kn":"2–3 ಸೆಂ.ಮೀ ನೀರಿನ ಮಟ್ಟ ಕಾಪಾಡಿ; ಸಸಿಗಳನ್ನು ರಕ್ಷಿಸಿ."},
              "tillering": {"en":"Apply urea (N) etc.","kn":"ಯೂರಿಯಾ (N) ನೀಡಿ; ಗಿಡ್ಮುಳ್ಳು ನಿಯಂತ್ರಿಸಿ."}},
    "maize": {"vegetative": {"en":"Apply nitrogen; maintain soil moisture.","kn":"ನೈಟ್ರೋಜನ್ ನೀಡಿ; ಮಣ್ಣು ತೇವ ಕಾಪಾಡಿ."}},
    # ... add others from previous list as needed
}

# fertilizer base simplified (use earlier full dictionary in your repo)
FERTILIZER_BASE = {
    "paddy": {"nursery": (20,10,10),"tillering": (60,30,20),"panicle initiation": (30,20,20)},
    "maize": {"vegetative": (80,40,20)},
    # ...
}

PESTICIDE_DB = {
    "aphid": {"en":"Spray neem oil (2%) or insecticidal soap.", "kn":"ನೀಮ್ ಎಣ್ಣೆ (2%) ಅಥವಾ ಸಾಬೂನು ಸಿಂಪಡಿಸಿ."},
    "whitefly": {"en":"Use yellow sticky traps, neem oil (2%).", "kn":"ಯೆಲ್ಲೋ ಸ್ಟಿಕ್ಕಿ ಟ್ರಾಪ್, ನೀಮ್ ಎಣ್ಣೆ."},
    "fruit borer": {"en":"Apply Bacillus thuringiensis (Bt).", "kn":"Bt ಬಳಸಿ."},
    # ... (extend as needed)
}

# Symptom DB for diagnose_advanced
SYMPTOM_DB = {
    "yellow leaves": ["nutrient deficiency", "nitrogen deficiency", "leaf curl virus", "wilt"],
    "leaf curling": ["leaf curl virus", "thrips", "aphid", "whitefly"],
    "white powder": ["powdery mildew"],
    "black spots": ["leaf spot", "early blight", "anthracnose"],
    # ... extend
}
SYMPTOM_SYNONYMS = {
    "yellowing": "yellow leaves", "leaf curl": "leaf curling", "white powdery": "white powder",
    # ...
}
CROP_SYMPTOM_WEIGHT = {
    "paddy": {"tungro": 2.0, "blast": 1.8},
    "tomato": {"late blight": 2.0, "early blight": 1.8},
    "chilli": {"fruit borer": 1.9},
    # ...
}
DISEASE_META = {
    "leaf curl virus": {"type": "viral", "note":"Usually transmitted by whiteflies"},
    "aphid": {"type":"insect","note":"Sucking insect - causes honeydew"},
    # ...
}

# -----------------------------
# Symptom matching & diagnosis functions (as previously)
# -----------------------------
def _extract_symptom_keys(user_text: str, fuzzy_threshold: float = 0.6):
    text = _normalize_text(user_text)
    found = []
    for phrase, canonical in sorted(SYMPTOM_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in text:
            found.append(canonical)
    for key in SYMPTOM_DB.keys():
        if key in text:
            found.append(key)
    tokens = _tokenize(text)
    joined = " ".join(tokens)
    for key in SYMPTOM_DB.keys():
        ratio = difflib.SequenceMatcher(None, joined, key).ratio()
        if ratio >= fuzzy_threshold:
            found.append(key)
    n = len(tokens)
    for L in range(2, min(6, n+1)):
        for i in range(n - L + 1):
            gram = " ".join(tokens[i:i+L])
            for phrase, canonical in SYMPTOM_SYNONYMS.items():
                if gram == phrase:
                    found.append(canonical)
    return list(dict.fromkeys(found))

def _score_candidates(symptom_keys: list, crop: Optional[str] = None):
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

def diagnose_advanced(user_text: str, user_crop: Optional[str] = None, lang: str = "en") -> Tuple[str, bool, list]:
    if not user_text or not user_text.strip():
        fallback = {"en":"Please describe the symptoms (leaf color, spots, pests seen, part affected).",
                    "kn":"ದಯವಿಟ್ಟು ಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ (ಎಲೆ ಬಣ್ಣ, ಕಲೆ, ಕಂಡ ಹಾಳುಕಾರಕಗಳು, ಭಾಗ ಪ್ರಭಾವಿತವಾಗಿರುವುದು)."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Describe symptoms"]
    symptom_keys = _extract_symptom_keys(user_text, fuzzy_threshold=0.58)
    if not symptom_keys:
        clauses = re.split(r"[,.;:/\\-]", user_text)
        for clause in clauses:
            keys = _extract_symptom_keys(clause, fuzzy_threshold=0.55)
            symptom_keys.extend(keys)
    symptom_keys = list(dict.fromkeys(symptom_keys))
    if not symptom_keys:
        fallback = {"en":"Couldn't identify clear symptoms. Please provide more details or upload a photo.",
                    "kn":"ನಿರ್ದಿಷ್ಟ ಲಕ್ಷಣಗಳು ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Contact Krishi Adhikari"]
    ranked = _score_candidates(symptom_keys, user_crop)
    if not ranked:
        fallback = {"en":"No candidate pests/diseases found for those symptoms.",
                    "kn":"ಆ ಲಕ್ಷಣಗಳಿಗೆ ಯೋಗ್ಯವಾದ ಕೀಟ/ರೋಗಗಳು ಕಂಡುಬರಲಿಲ್ಲ."}
        return fallback.get(lang, fallback["en"]), False, ["Upload photo", "Contact Krishi Adhikari"]
    top_k = ranked[:3]
    lines = []
    header = "Likely pests/diseases (top candidates):\n" if lang != "kn" else "ಸರಾಸರಿ ಅನುಮಾನಿತ ರೋಗ/ಕೀಟಗಳು (ಮೇಲವರ್ಗ):\n"
    lines.append(header)
    for cand, score, conf, ev in top_k:
        meta = DISEASE_META.get(cand, {})
        meta_note = meta.get("note", "")
        lines.append(f"- {cand.title()} (confidence: {int(conf*100)}%)")
        if meta_note:
            lines.append(f"    • {meta_note}")
        lines.append(f"    • Evidence: {', '.join(ev)}")
    # pesticide recommendations if available
    rec_texts = []
    for cand, _, _, _ in top_k:
        key = cand.lower()
        if key in PESTICIDE_DB:
            rec = PESTICIDE_DB[key].get(lang if lang in ["en","kn"] else "en")
            if rec:
                rec_texts.append(f"For {cand.title()}: {rec}")
    if rec_texts:
        lines.append("\nSuggested interventions:")
        for r in rec_texts:
            lines.append(f"- {r}")
    lines.append("\nIdentified symptoms:")
    for s in symptom_keys:
        lines.append(f"- {s}")
    final_text = "\n".join(lines)
    return final_text, False, ["Upload photo", "Pesticide recommendations", "Contact advisor"]

# -----------------------------
# Soil testing center, pest_disease quick checks, farm_timeline, get_latest_crop_stage
# -----------------------------
def soil_testing_center(user_id: str, language: str):
    loc = get_user_location(user_id)
    if not loc:
        msg = {"en":"Farm location not found. Update district & taluk in farmDetails.",
               "kn":"ಫಾರಂ ಸ್ಥಳದ ಮಾಹಿತಿ ಕಂಡುಬರಲಿಲ್ಲ. farmDetails ನಲ್ಲಿ ಜಿಲ್ಲೆ ಮತ್ತು ತಾಲೂಕು ನವೀಕರಿಸಿ."}
        return msg[language], True, ["Update farm details"]
    district, taluk = loc.get("district"), loc.get("taluk")
    if not district or not taluk:
        return ("Farm district/taluk missing in farmDetails.", True, ["Update farm details"])
    centers = firebase_get(f"SoilTestingCenters/Karnataka/{district}/{taluk}")
    if not centers:
        return ("No soil test center found for your area.", True, ["Update farm details"])
    for _, info in centers.items():
        if isinstance(info, dict):
            text = f"{info.get('name')}\n{info.get('address')}\nContact: {info.get('contact')}"
            return text, True, ["Directions", "Call center"]
    return "No center data available.", True, []

def pest_disease(query: str, language: str):
    q = (query or "").lower()
    if "curl" in q or "curled" in q:
        en = "Symptoms indicate leaf curl virus or sucking pests. Remove severely affected shoots and apply neem oil spray."
        kn = "ಎಲೆ ಕರ್ಭಟ ವೈರಸ್ ಅಥವಾ ಸ್ಯಕ್ಕಿಂಗ್ ಕೀಟಗಳ ಸೂಚನೆ. ಗಂಭೀರವಾದ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನೀಮ್ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ."
        return (kn if language == "kn" else en), True, ["Neem spray", "Contact Krishi Adhikari"]
    if "yellow" in q or "yellowing" in q:
        en = "Yellowing leaves may indicate nutrient deficiency or overwatering. Check soil moisture and consider soil test."
        kn = "ಎಲೆಗಳು ಹಳದಿ ಆಗುವುದು ಪೋಷಕಾಂಶ ಕೊರತೆ ಅಥವಾ ಹೆಚ್ಚಾಗಿ ನೀರು ಕಾರಣವಾಗಬಹುದು."
        return (kn if language == "kn" else en), True, ["Soil test", "Nitrogen application"]
    fallback = {"en":"Provide more symptom details or upload a photo.", "kn":"ಲಕ್ಷಣಗಳ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ವಿವರ ನೀಡಿ ಅಥವಾ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ."}
    return fallback[language], True, ["Upload photo"]

def farm_timeline(user_id: str, language: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return ("No activity logs found." if language == "en" else "ಚಟುವಟಿಕೆ ಲಾಗ್ ಕಂಡುಬರಲಿಲ್ಲ."), False, ["Add activity"]
    summaries = []
    for crop, entries in (logs.items() if isinstance(logs, dict) else []):
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

def get_latest_crop_stage(user_id: str, lang: str = "en"):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs")
    if not logs:
        return ("No farm activity found." if lang == "en" else "ಫಾರಂ ಚಟುವಟಿಕೆ ಕಂಡುಬರಲಿಲ್ಲ."), False, ["Add activity"]
    latest_ts = -1
    latest_crop = None
    latest_stage = None
    for crop, entries in (logs.items() if isinstance(logs, dict) else []):
        if isinstance(entries, dict):
            for act_id, data in entries.items():
                ts = data.get("timestamp", 0)
                if ts and ts > latest_ts:
                    latest_ts = ts
                    latest_crop = data.get("cropName", crop)
                    latest_stage = data.get("stage", "Unknown")
    rec = STAGE_RECOMMENDATIONS.get((latest_crop or "").lower(), {}).get((latest_stage or "").lower())
    if rec:
        text = rec.get(lang if lang in ["en","kn"] else "en")
    else:
        text = STAGE_RECOMMENDATIONS.get((latest_crop or "").lower(), {}).get("vegetative", {}).get(lang, "") or \
               f"No specific recommendation for {latest_crop} at stage '{latest_stage}'."
    header = (f"{latest_crop} ಬೆಳೆ ಪ್ರಸ್ತುತ ಹಂತ: {latest_stage}\n\n" if lang == "kn" else f"Current stage of {latest_crop}: {latest_stage}\n\n")
    return header + text, False, ["Next actions", "Fertilizer advice", "Pest check"]

# -----------------------------
# Weather / irrigation modules (mock + fetch)
# -----------------------------
def fetch_weather_by_location(district: str):
    """Fetch from OpenWeather; fallback to None on error"""
    if not OPENWEATHER_KEY or not district:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={district}&appid={OPENWEATHER_KEY}&units=metric"
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
    except Exception:
        return None

def get_mock_weather_for_district(district):
    return {"temp": 30, "humidity": 70, "wind": 8, "rain_next_24h_mm": 0, "rain": 0, "condition":"Clear", "description":"clear sky"}

def weather_suggestion_engine(weather, crop_stage=None, language="en"):
    if not weather:
        return []
    temp = weather.get("temp", 30)
    humidity = weather.get("humidity", 60)
    wind = weather.get("wind", 5)
    rain = weather.get("rain", 0)
    cond = weather.get("condition", "")

    suggestions = []
    if temp > 35:
        suggestions.append("High heat – give afternoon irrigation and mulch.")
    elif temp < 15:
        suggestions.append("Low temperature – avoid fertilizer today.")
    if rain > 3:
        suggestions.append("Rainfall occurring – stop irrigation for 24 hours.")
    else:
        suggestions.append("No rain – irrigation recommended today.")
    if humidity > 80:
        suggestions.append("High humidity – fungal disease chances are high.")
    elif humidity < 35:
        suggestions.append("Low humidity – increase irrigation frequency.")
    if wind > 20:
        suggestions.append("High wind – avoid spraying pesticides.")
    if crop_stage:
        st = crop_stage.lower()
        if "flower" in st and cond == "Rain":
            suggestions.append("Rain during flowering – flower drop likely.")
        if "harvest" in st and rain > 0:
            suggestions.append("Rain coming – postpone harvest.")
    # Translate to Kannada if requested - minimal mapping
    if language == "kn":
        mapping = {
            "High heat – give afternoon irrigation and mulch.":"ಹೆಚ್ಚು ಬಿಸಿಲು – ಮಧ್ಯಾಹ್ನ ನೀರಾವರಿ ಮಾಡಿ ಮತ್ತು ಮಲ್ಚಿಂಗ್ ಮಾಡಿ.",
            "Low temperature – avoid fertilizer today.":"ಕಡಿಮೆ ತಾಪಮಾನ – ಇಂದು ರಸಗೊಬ್ಬರ ಬಳಕೆ ಬೇಡ.",
            "Rainfall occurring – stop irrigation for 24 hours.":"ಮಳೆ ಬರುತ್ತಿದೆ – 24 ಗಂಟೆಗಳ ಕಾಲ ನೀರಾವರಿ ನಿಲ್ಲಿಸಿ.",
            "No rain – irrigation recommended today.":"ಮಳೆಯಿಲ್ಲ – ಇಂದು ನೀರಾವರಿ ಮಾಡಿರಿ.",
            "High humidity – fungal disease chances are high.":"ಹೆಚ್ಚು ತೇವಾಂಶ – ಫಂಗಸ್ ರೋಗದ ಸಾಧ್ಯತೆ ಹೆಚ್ಚು.",
            "Low humidity – increase irrigation frequency.":"ಕಡಿಮೆ ತೇವಾಂಶ – ನೀರಾವರಿ ಪ್ರಮಾಣ ಹೆಚ್ಚಿಸಿ.",
            "High wind – avoid spraying pesticides.":"ಬಲವಾದ ಗాలి – ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ ಬೇಡ.",
            "Rain during flowering – flower drop likely.":"ಹೂ ಹಂತದಲ್ಲಿ ಮಳೆ – ಹೂ ಬಿದ್ದು ಹೋಗುವ ಸಾಧ್ಯತೆ.",
            "Rain coming – postpone harvest.":"ಮಳೆ ಬರಲಿದೆ – ಕೊಯ್ತನ್ನು ಮುಂದೂಡಿ."
        }
        suggestions = [mapping.get(s, s) for s in suggestions]
    return suggestions

# -----------------------------
# Route / intent engine
# -----------------------------
def get_prompt(lang: str) -> str:
    return f"You are KrishiSakhi. Respond only in {'Kannada' if lang == 'kn' else 'English'} with short actionable crop advice."

def crop_advisory(user_id: str, query: str, lang: str, session_key: str):
    """
    Uses Hugging Face Mixtral generate to create crop advisory.
    Maintains a very simple session memory (last prompt) if desired in active_chats.
    """
    global active_chats
    if not HUGGINGFACE_API_KEY:
        return "AI (HuggingFace) not configured on server.", False, [], session_key
    # Build small system style prompt to keep responses short and actionable
    prompt = get_prompt(lang) + "\n\nFarmer query: " + query + "\n\nReply concisely:"
    text, err = hf_generate(prompt, model=HUGGINGFACE_MODEL, max_new_tokens=256, temperature=0.2)
    if err:
        logger.warning("HF generate error: %s", err)
        return f"AI error: {err}", False, [], session_key
    if not text:
        return "AI returned empty response.", False, [], session_key
    # Optionally trim excessive output
    return text.strip(), False, ["Crop stage", "Pest check", "Soil test"], session_key

def route(query: str, user_id: str, lang: str, session_key: str):
    q = (query or "").lower().strip()
    # Intent checks
    if any(tok in q for tok in ["soil test", "soil testing", "soil centre", "soil center"]):
        return {"response_text": soil_testing_center(user_id, lang)[0], "voice": True, "suggestions": ["Update farm details"]}
    if any(tok in q for tok in ["timeline", "activity log", "farm activity"]):
        t, v, s = farm_timeline(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["weather", "rain", "forecast"]) and "stage" not in q:
        report, sug, voice = weather_advisory(user_id, lang) if 'weather_advisory' in globals() else ("Weather module disabled", [], False)
        return {"response_text": report, "voice": voice, "suggestions": sug}
    if any(tok in q for tok in ["price", "market", "mandi"]):
        t, v, s = market_price(query, lang) if 'market_price' in globals() else ("Market module disabled", False, [])
        return {"response_text": t, "voice": v, "suggestions": s}
    if "crop stage" in q or q == "stage" or "stage" in q:
        t, v, s = get_latest_crop_stage(user_id, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if any(tok in q for tok in ["pest", "disease", "leaf", "spots", "yellowing", "curl", "blight", "fungus"]):
        # Try symptom-based diagnosis first
        diag_text, voice, sugg = diagnose_advanced(query, user_crop=None, lang=lang)
        return {"response_text": diag_text, "voice": voice, "suggestions": sugg}
    if "fertilizer" in q or "fertiliser" in q or "apply fertilizer" in q:
        # attempt to pick farm's latest crop/stage
        ret = fertilizer_intent_handler(user_id, lang)
        return ret
    if "pesticide" in q or "spray" in q or "aphid" in q or "fruit borer" in q:
        pest = None
        for key in PESTICIDE_DB.keys():
            if key in q:
                pest = key
                break
        if not pest:
            msg = ("Please tell me the pest name or upload a photo (e.g., 'aphid')." if lang == "en"
                   else "ದಯವಿಟ್ಟು ಕೀಟದ ಹೆಸರು ಅಥವಾ ಫೋಟೋ ನೀಡಿ (ಉದಾ: aphid).")
            return {"response_text": msg, "voice": False, "suggestions": ["Upload photo", "aphid"]}
        t, v, s = pesticide_recommendation("", pest, lang)
        return {"response_text": t, "voice": v, "suggestions": s}
    if "irrigation" in q or "water" in q or "irrigate" in q:
        ret = irrigation_intent_handler(user_id, lang)
        return ret
    if "yield" in q or "estimate" in q or "production" in q:
        ret = yield_intent_handler(user_id, q, lang)
        return ret
    if "weather" in q and "stage" in q:
        # weather + stage fusion
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop=None; latest_stage=None; latest_ts=-1
        for crop, entries in (logs.items() if isinstance(logs, dict) else []):
            if isinstance(entries, dict):
                for _, data in entries.items():
                    ts = data.get("timestamp", 0)
                    if ts and ts > latest_ts:
                        latest_ts = ts
                        latest_crop = data.get("cropName", crop)
                        latest_stage = data.get("stage", "")
        if not latest_crop:
            return {"response_text":"No crop found. Add crop activity.","voice":False,"suggestions":["Add activity"]}
        text, v, s = weather_crop_fusion(user_id, latest_crop, latest_stage, lang)
        return {"response_text": text, "voice": v, "suggestions": s}
    # Default -> HF crop advisory fallback
    t, v, s, sid = crop_advisory(user_id, query, lang, session_key)
    return {"response_text": t, "voice": v, "suggestions": s, "session_id": sid}

# -----------------------------
# Small helper handlers for fertilizer/irrigation/yield/pesticide to keep route compact
# (Implementations are simplified copies of earlier logic)
# -----------------------------
def fertilizer_intent_handler(user_id: str, lang: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
    latest_crop=None; latest_stage=None; latest_ts=-1
    for crop, entries in (logs.items() if isinstance(logs, dict) else []):
        if isinstance(entries, dict):
            for aid, data in entries.items():
                ts = data.get("timestamp", 0)
                if ts and ts > latest_ts:
                    latest_ts = ts
                    latest_crop = data.get("cropName", crop)
                    latest_stage = data.get("stage", "")
    if not latest_crop:
        msg = ("Please provide the crop and stage (e.g., 'fertilizer for paddy tillering')"
               if lang == "en" else "ದಯವಿಟ್ಟು ಬೆಳೆ ಮತ್ತು ಹಂತವನ್ನು ನೀಡಿ (ಉದಾ: 'fertilizer for paddy tillering')")
        return {"response_text": msg, "voice": False, "suggestions": ["Provide crop & stage"]}
    t, v, s = fertilizer_calculator(latest_crop, latest_stage, user_id, lang)
    return {"response_text": t, "voice": v, "suggestions": s}

def irrigation_intent_handler(user_id: str, lang: str):
    logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
    latest_crop=None; latest_stage=None; latest_ts=-1
    for crop, entries in (logs.items() if isinstance(logs, dict) else []):
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

def yield_intent_handler(user_id: str, query: str, lang: str):
    crop = None
    for c in list(STAGE_RECOMMENDATIONS.keys()) + list(BASE_YIELD_TON_PER_HA.keys() if 'BASE_YIELD_TON_PER_HA' in globals() else []):
        if c in query:
            crop = c
            break
    if not crop:
        logs = firebase_get(f"Users/{user_id}/farmActivityLogs") or {}
        latest_crop = None; latest_ts=-1
        for crop_k, entries in (logs.items() if isinstance(logs, dict) else []):
            if isinstance(entries, dict):
                for aid, data in entries.items():
                    ts = data.get("timestamp", 0)
                    if ts and ts > latest_ts:
                        latest_ts = ts
                        latest_crop = data.get("cropName", crop_k)
        crop = latest_crop or "paddy"
    t, v, s = yield_prediction(crop, user_id, lang)
    return {"response_text": t, "voice": v, "suggestions": s}

# -----------------------------
# Implementations copied/linked from earlier modules:
# fertilizer_calculator, pesticide_recommendation, irrigation_schedule, yield_prediction, weather_crop_fusion
# (We include minimal working versions here — replace or expand as needed)
# -----------------------------
def fertilizer_calculator(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    area_ha = None
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
            text = (f"{crop.title()} - {stage.title()} ಹಂತಕ್ಕೆ ಶಿಫಾರಸು (ಪ್ರದೇಶ {area_ha} ha):\nN: {N} kg, P2O5: {P} kg, K2O: {K} kg.")
        else:
            text = (f"Fertilizer recommendation for {crop.title()} ({stage.title()}) for {area_ha} ha:\nN: {N} kg, P2O5: {P} kg, K2O: {K} kg.")
        return text, False, ["Soil test", "Buy fertilizer"]
    else:
        fallback = {"en":"No fertilizer template available for this crop/stage. Provide crop and stage or run soil test.",
                    "kn":"ಈ ಬೆಳೆ/ಹಂತಕ್ಕೆ ರೂಪರೆಖೆ ಲಭ್ಯವಿಲ್ಲ. ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮಾಡಿ."}
        return fallback[lang], False, ["Soil test"]

def pesticide_recommendation(crop: str, pest: str, lang: str) -> Tuple[str, bool, List[str]]:
    pest_l = (pest or "").lower()
    if pest_l in PESTICIDE_DB:
        return PESTICIDE_DB[pest_l][lang if lang in ["en","kn"] else "en"], False, ["Use bio-pesticide", "Contact advisor"]
    for key in PESTICIDE_DB.keys():
        if key in pest_l:
            return PESTICIDE_DB[key][lang if lang in ["en","kn"] else "en"], False, ["Use bio-pesticide", "Contact advisor"]
    fallback = {"en":"Pest not recognized. Provide photo or pest name (e.g., 'aphid').","kn":"ಕೀಟ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ಫೋಟೋ ನೀಡಿ."}
    return fallback[lang], False, ["Upload photo", "Contact Krishi Adhikari"]

# irrigation schedule (simplified)
SOIL_WATER_HOLDING = {"sandy":0.6,"loamy":1.0,"clay":1.2}
CROP_ET_BASE = {"paddy":6.0,"maize":5.5,"tomato":4.8}
def irrigation_schedule(crop: str, stage: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    soil = (farm.get("soilType") or "loamy").lower() if isinstance(farm, dict) else "loamy"
    try:
        area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0) if isinstance(farm, dict) else 1.0
    except Exception:
        area_ha = 1.0
    district = farm.get("district") if isinstance(farm, dict) else "unknown"
    weather = get_mock_weather_for_district(district)
    rain_next_24 = weather.get("rain_next_24h_mm", 0)
    crop_l = (crop or "").lower()
    base_et = CROP_ET_BASE.get(crop_l, 4)
    soil_factor = SOIL_WATER_HOLDING.get(soil, 1.0)
    stage_mult = 1.0
    st = (stage or "").lower()
    if "nursery" in st or "vegetative" in st:
        stage_mult = 1.2
    elif "flower" in st:
        stage_mult = 1.1
    elif "harvest" in st:
        stage_mult = 0.8
    required_mm = base_et * stage_mult * (1.0 / soil_factor)
    if rain_next_24 >= 10:
        suggestion = {"en":"Rain expected soon. Delay irrigation and monitor soil moisture.","kn":"ಶೀಘ್ರದಲ್ಲೇ ಮಳೆಯ ಸಂಭವನೆ. ನೀರಾವರಿ ತಡೆಯಿರಿ."}
        return suggestion[lang], False, ["Soil moisture check", "Delay irrigation"]
    liters_per_ha = required_mm * 10000
    total_liters = round(liters_per_ha * area_ha, 1)
    if lang == "kn":
        text = f"{crop.title()} ({stage}) - ಪ್ರತಿ ದಿನ {round(required_mm,1)} mm (~{total_liters} L/day for {area_ha} ha)."
    else:
        text = f"Recommendation for {crop.title()} ({stage}): approx {round(required_mm,1)} mm/day (~{total_liters} liters/day for {area_ha} ha)."
    return text, False, ["Soil moisture sensor", "Irrigation logs"]

# yield prediction (simplified)
BASE_YIELD_TON_PER_HA = {"paddy":4.0,"maize":3.5,"tomato":25.0}
def yield_prediction(crop: str, user_id: str, lang: str) -> Tuple[str, bool, List[str]]:
    farm = get_user_farm_details(user_id)
    try:
        area_ha = float(farm.get("areaInHectares") or farm.get("area") or 1.0) if isinstance(farm, dict) else 1.0
    except Exception:
        area_ha = 1.0
    crop_l = (crop or "").lower()
    base = BASE_YIELD_TON_PER_HA.get(crop_l, 2.0)
    last_fert = firebase_get(f"Users/{user_id}/lastFertilizerApplied") or {}
    fert_ok = isinstance(last_fert, dict) and last_fert.get("applied", False)
    irrigation_logs = firebase_get(f"Users/{user_id}/irrigationLogs") or {}
    irrigation_ok = False
    if isinstance(irrigation_logs, dict):
        found_recent = False
        now = datetime.utcnow().timestamp()
        for _, v in irrigation_logs.items():
            ts = v.get("timestamp", 0)
            if now - ts < 14*24*3600:
                found_recent = True; break
        irrigation_ok = found_recent
    pest_incidents = firebase_get(f"Users/{user_id}/pestIncidents") or {}
    pest_control_ok = not (isinstance(pest_incidents, dict) and len(pest_incidents) > 0)
    fert_factor = 1.1 if fert_ok else 0.9
    irr_factor = 1.05 if irrigation_ok else 0.9
    pest_factor = 0.95 if not pest_control_ok else 1.0
    predicted_ton_per_ha = round(base * fert_factor * irr_factor * pest_factor, 2)
    total_tonnage = round(predicted_ton_per_ha * area_ha, 2)
    if lang == "kn":
        text = f"ಅಂದಾಜು ಉತ್ಪಾದನೆ: {predicted_ton_per_ha} ಟನ್/ha. ಒಟ್ಟು ~{total_tonnage} ಟನ್ ({area_ha} ha)."
    else:
        text = f"Estimated yield: {predicted_ton_per_ha} ton/ha. Total ~{total_tonnage} ton for {area_ha} ha."
    return text, False, ["Improve irrigation", "Soil test", "Pest control"]

def weather_crop_fusion(user_id: str, crop: str, stage: str, lang: str):
    farm = get_user_farm_details(user_id)
    district = farm.get("district", "unknown")
    weather = fetch_weather_by_location(district) or get_mock_weather_for_district(district)
    if not weather:
        return ("Weather data unavailable.", False, ["Retry"])
    stage_advice = STAGE_RECOMMENDATIONS.get((crop or "").lower(), {}).get((stage or "").lower(), {}).get(lang, "")
    fusion = weather_suggestion_engine(weather, crop_stage=stage, language=lang)
    if lang == "kn":
        report = (f"{district} ಹವಾಮಾನ:\nತಾಪಮಾನ: {weather['temp']}°C | ತೇವಾಂಶ: {weather['humidity']}%\nಹಂತ: {crop} – {stage}\n\nಹಂತ ಸಲಹೆ:\n{stage_advice}\n\nಹವಾಮಾನ ಆಧಾರಿತ ಹೆಚ್ಚುವರಿ ಸಲಹೆಗಳು:\n- " + "\n- ".join(fusion))
    else:
        report = (f"Weather in {district}:\nTemp: {weather['temp']}°C | Humidity: {weather['humidity']}%\nStage: {crop} – {stage}\n\nStage Recommendation:\n{stage_advice}\n\nWeather-based Additional Advice:\n- " + "\n- ".join(fusion))
    return report, False, ["Fertilizer", "Pest Check", "Irrigation"]

# -----------------------------
# Market price lightweight function (uses PRICE_LIST)
# -----------------------------
def market_price(query: str, language: str):
    q = (query or "").lower()
    for crop, price in PRICE_LIST.items():
        if crop in q:
            if language == "kn":
                return f"{crop.title()} ಸರಾಸರಿ ಬೆಲೆ: ₹{price}/ಕಿ.ಗ್ರಾಂ. ಸ್ಥಳೀಯ APMC ಪರಿಶೀಲಿಸಿ.", False, ["Sell at APMC", "Quality Check"]
            return f"Approx price for {crop.title()}: ₹{price}/kg. Verify with local APMC.", False, ["Sell at APMC", "Quality Check"]
    fallback = {"en":"Please specify the crop name (e.g., 'chilli price').","kn":"ದಯವಿಟ್ಟು ಬೆಳೆ ಹೆಸರು ನೀಡಿ (ಉದಾ: 'ಮೆಣಸಿನ ಬೆಲೆ')."}
    return fallback[language], False, ["Chilli price", "Areca price"]

# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/chat/send", response_model=ChatResponse)
async def chat_send(payload: ChatQuery):
    user_query = (payload.user_query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    lang = get_language(payload.user_id)
    session_key = payload.session_id or f"{payload.user_id}-{lang}"
    try:
        result = route(user_query, payload.user_id, lang, session_key)
    except Exception as e:
        logger.exception("Processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    # TTS attempt
    audio_url = None
    try:
        if result.get("response_text"):
            audio_url = generate_tts_audio(result["response_text"], lang)
    except Exception as e:
        logger.exception("TTS generation failed: %s", e)
        audio_url = None
    return ChatResponse(
        session_id=result.get("session_id", session_key),
        response_text=result.get("response_text", "Sorry, could not process."),
        language=lang,
        suggestions=result.get("suggestions", []),
        voice=True,
        audio_url=audio_url,
        metadata={"timestamp": datetime.utcnow().isoformat()}
    )

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
def startup():
    logger.info("Starting KS Backend (HF) ...")
    # ensure tts folder exists
    os.makedirs(TTS_DIR, exist_ok=True)
    # init firebase credentials (if provided)
    initialize_firebase_credentials()
    logger.info("Startup complete.")

