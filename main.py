"""
NADRA Multilingual RAG Assistant — FastAPI Backend
Integrated with Simli Avatar API (WebRTC via /compose/token)
"""

import os
import tempfile
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import re
import io
import base64
import asyncio
import requests
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Tuple

import torch
import chromadb
import edge_tts
from pydub import AudioSegment
from groq import Groq
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, CrossEncoder
from faster_whisper import WhisperModel

# ============================================================
# CONFIGURATION
# ============================================================

import config

GROQ_API_KEY        = config.GROQ_API_KEY
SIMLI_API_KEY       = config.SIMLI_API_KEY
SIMLI_FACE_ID       = config.SIMLI_FACE_ID
SIMLI_TOKEN_URL     = config.SIMLI_TOKEN_URL
SIMLI_API_VERSION   = config.SIMLI_API_VERSION
CHROMA_PERSIST_DIR  = config.CHROMA_PERSIST_DIR

GROQ_MODEL         = config.GROQ_MODEL
EMBED_MODEL_NAME   = config.EMBED_MODEL_NAME
RERANK_MODEL_NAME  = config.RERANK_MODEL_NAME

VOICE_EN           = config.VOICE_EN
VOICE_UR           = config.VOICE_UR

TOP_K              = config.TOP_K
FINAL_K            = config.FINAL_K
DetectorFactory.seed = 0

# Languages Whisper may return that should be converted to Urdu
CONVERT_TO_UR_LANGS = config.CONVERT_TO_UR_LANGS

SYSTEM_PROMPT = config.SYSTEM_PROMPT

print(f"[CONFIG] SIMLI_API_KEY = '{SIMLI_API_KEY}'")
print(f"[CONFIG] SIMLI_FACE_ID = '{SIMLI_FACE_ID}'")

# ============================================================
# Load Models
# ============================================================

print("Loading models...")

device       = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"

embedder      = SentenceTransformer(EMBED_MODEL_NAME)
reranker      = CrossEncoder(RERANK_MODEL_NAME, device=device)
whisper_model = WhisperModel(config.WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
groq_client   = Groq(api_key=GROQ_API_KEY)

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
COLLECTIONS = {
    lang: chroma_client.get_collection(name)
    for lang, name in config.CHROMA_COLLECTIONS.items()
}

print("All models loaded.")

# ============================================================
# Utility Functions
# ============================================================
# NOTE: SYSTEM_PROMPT is defined in config.py and imported above.

def detect_language(text: str) -> str:
    """Detect language from text. Returns 'ur' or 'en'."""
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "ur"
    try:
        lang = detect(text)
        if lang in ["ur", "ar", "fa"]:
            return "ur"
    except:
        pass
    return "en"


async def normalize_to_urdu(text: str) -> str:
    """
    Use Groq to convert Hindi / Arabic / other transcription into Pakistani Urdu.
    Called when Whisper detects a language in CONVERT_TO_UR_LANGS.
    """
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a language converter. The user will provide text in Hindi, Arabic, "
                    "or a similar language. Rewrite it in Pakistani Urdu script only — do NOT "
                    "translate the meaning, just convert the script and vocabulary to Pakistani Urdu. "
                    "Return ONLY the converted Urdu text, nothing else."
                )
            },
            {"role": "user", "content": text}
        ],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def retrieve_and_rerank(query: str, lang: str) -> str:
    collection = COLLECTIONS.get(lang)
    if not collection:
        return ""
    query_emb = embedder.encode(f"query: {query}", normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=TOP_K)
    documents = results["documents"][0]
    if not documents:
        return ""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    if not hasattr(scores, '__len__'):
        scores = [scores]
    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return "\n\n".join([doc for _, doc in scored[:FINAL_K]])


def enhance_query(raw_text: str, history: List[Tuple[str, str]]) -> str:
    context_snippet = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in history[-2:]])
    prompt = (
        f"Clean and optimize this query for NADRA document search.\n"
        f"Query: {raw_text}\nHistory: {context_snippet}\n"
        f"Return only the cleaned query, nothing else."
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a query optimizer. Return only the cleaned query."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def call_groq(context: str, user_text: str, lang: str, history: List[Tuple[str, str]]) -> str:
    lang_note = (
        "\n\nIMPORTANT: Respond in ENGLISH ONLY." if lang == "en"
        else "\n\nIMPORTANT: Respond in PAKISTANI URDU ONLY. No Hindi. No Devanagari."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + lang_note}]
    for h_user, h_bot in history[-5:]:
        clean_bot = re.sub('<[^<]+?>', '', h_bot)
        messages.append({"role": "user", "content": h_user})
        messages.append({"role": "assistant", "content": clean_bot})
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{user_text}"})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# ============================================================
# TTS — returns PCM16 mono 16kHz (Simli's required format)
# ============================================================

async def generate_tts(text: str, primary_lang: str) -> bytes:
    """Generate TTS and convert to PCM16 mono 16kHz for Simli WebRTC."""
    try:
        from gtts import gTTS
        lang_code = "ur" if primary_lang == "ur" else "en"
        tts = gTTS(text=text, lang=lang_code, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        audio = AudioSegment.from_mp3(mp3_buf)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        pcm_buf = io.BytesIO()
        audio.export(pcm_buf, format="raw")
        return pcm_buf.getvalue()

    except Exception as e:
        print(f"TTS error: {e}")
        return b""

# ============================================================
# Simli — graceful fallback if unreachable
# ============================================================

def get_simli_session_token() -> Optional[str]:
    """
    Returns a session_token string, or None if Simli is unavailable.
    The app continues working in text-only mode if None is returned.
    """
    print(f"[SIMLI] Requesting token with key='{SIMLI_API_KEY}' faceId='{SIMLI_FACE_ID}'")
    try:
        res = requests.post(
            SIMLI_TOKEN_URL,
            headers={
                "x-simli-api-key": SIMLI_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "faceId": SIMLI_FACE_ID,
                "apiVersion": SIMLI_API_VERSION,
                "handleSilence": True,
                "maxSessionLength": config.SIMLI_MAX_SESSION_LENGTH,
                "maxIdleTime": config.SIMLI_MAX_IDLE_TIME,
                "audioInputFormat": config.SIMLI_AUDIO_INPUT_FORMAT
            },
            timeout=config.SIMLI_REQUEST_TIMEOUT
        )
        print(f"[SIMLI] HTTP {res.status_code} — {res.text[:200]}")
        res.raise_for_status()
        data = res.json()
        token = data.get("session_token")
        if not token:
            print(f"[SIMLI] No session_token in response: {data}")
            return None
        print(f"[SIMLI] Token obtained successfully.")
        return token
    except Exception as e:
        print(f"[SIMLI] Unavailable (non-fatal): {e}")
        return None

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title=config.APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    history: List[List[str]] = []
    simli_session_token: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    simli_session_token: str
    detected_lang: str
    audio_b64: Optional[str] = None
    simli_available: bool = False


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/start-session")
async def start_session():
    """
    Called once by the frontend on page load.
    Always returns 200 — simli_available tells the frontend whether avatar is usable.
    """
    token = get_simli_session_token()
    return {
        "simli_session_token": token or "",
        "simli_available": token is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        print(f"[DEBUG] req.text={req.text!r}")
        input_lang    = detect_language(req.text)
        print(f"[DEBUG] input_lang={input_lang}")

        history_pairs = [
            (str(h[0]), str(h[1]))
            for h in req.history
            if isinstance(h, (list, tuple)) and len(h) == 2
        ]

        enhanced  = enhance_query(req.text, history_pairs)
        context   = retrieve_and_rerank(enhanced, input_lang)
        answer    = call_groq(context, req.text, input_lang, history_pairs)
        ans_lang  = detect_language(answer)
        print(f"[DEBUG] answer={answer!r}")

        audio_bytes = await generate_tts(answer, ans_lang)

        # Reuse existing token if frontend already has one, otherwise fetch fresh
        session_token = req.simli_session_token or get_simli_session_token()

        return ChatResponse(
            answer=answer,
            simli_session_token=session_token or "",
            detected_lang=ans_lang,
            audio_b64=base64.b64encode(audio_bytes).decode() if audio_bytes else None,
            simli_available=bool(session_token)
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Chat error: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        tmp_path = os.path.join(tempfile.gettempdir(), audio.filename)
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            initial_prompt="CNIC, B-Form, NADRA, NICOP, FRC, Smart Card"
        )
        text = " ".join([s.text for s in segments]).strip()
        os.remove(tmp_path)

        # Use Whisper's own language detection — far more reliable than langdetect on short audio
        detected = info.language  # e.g. "hi", "ur", "en", "ar", "fa", "pa"
        print(f"[TRANSCRIBE] Whisper detected language: '{detected}' | text: {text!r}")

        if detected in CONVERT_TO_UR_LANGS:
            # Hindi / Arabic / Farsi / Punjabi → convert to Pakistani Urdu via Groq
            print(f"[TRANSCRIBE] Converting '{detected}' → Urdu via Groq")
            text = await normalize_to_urdu(text)
            lang = "ur"

        elif detected == "ur" or any("\u0600" <= c <= "\u06FF" for c in text):
            # Already Urdu
            lang = "ur"

        elif detected == "en":
            # English — pass through as-is
            lang = "en"

        else:
            # Unsupported language — return a fallback message, skip /chat entirely
            print(f"[TRANSCRIBE] Unsupported language '{detected}' — returning fallback")
            return JSONResponse(content={
                "text": "I cannot understand you. Please speak in Urdu or English.",
                "lang": "en",
                "unsupported": True
            })

        return JSONResponse(content={
            "text": text,
            "lang": lang,
            "unsupported": False
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Transcribe error: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})


@app.get("/health")
async def health():
    return {"status": "ok"}
