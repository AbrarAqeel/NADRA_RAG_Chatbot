"""
Configuration for the NADRA Multilingual RAG Assistant.

- Secrets and environment-specific values are loaded from .env.
- Everything else (models, retrieval params, prompts, etc.) is defined here.
- main.py should import this module and never hardcode a configurable value.

If a variable is missing from .env, we fall back to the default shown below
(silent fallback — no error is raised).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Secrets & environment-specific values (from .env)
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "")

SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "0c2b8b04-5274-41f1-a21c-d5c98322efa9")
SIMLI_TOKEN_URL = os.getenv("SIMLI_TOKEN_URL", "https://api.simli.ai/compose/token")
SIMLI_API_VERSION = os.getenv("SIMLI_API_VERSION", "v2")

# ============================================================
# App
# ============================================================
APP_TITLE = "NADRA Simli Assistant"

# ============================================================
# Simli Avatar (non-environment-specific settings)
# ============================================================
SIMLI_MAX_SESSION_LENGTH = 3600     # seconds
SIMLI_MAX_IDLE_TIME = 300           # seconds
SIMLI_AUDIO_INPUT_FORMAT = "pcm16"
SIMLI_REQUEST_TIMEOUT = 15          # seconds

# ============================================================
# Chroma / Vector DB
# ============================================================
CHROMA_PERSIST_DIR = "/app/chroma_db_fixed"
CHROMA_COLLECTIONS = {
    "en": "nadra_en",
    "ur": "nadra_ur",
}

# ============================================================
# Groq LLM
# ============================================================
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================================
# Embedding / Reranking Models
# ============================================================
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================
# Whisper (Speech-to-Text)
# ============================================================
WHISPER_MODEL_SIZE = "large-v3"

# ============================================================
# Text-to-Speech Voices
# ============================================================
VOICE_EN = "en-US-AriaNeural"
VOICE_UR = "ur-PK-UzmaNeural"

# ============================================================
# Retrieval
# ============================================================
TOP_K = 15     # candidates fetched from vector store
FINAL_K = 5    # candidates kept after reranking

# ============================================================
# Language Handling
# ============================================================
# Languages Whisper may return that should be converted to Urdu
CONVERT_TO_UR_LANGS = {"hi", "ar", "fa", "pa"}  # Hindi, Arabic, Farsi, Punjabi

# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """
You are the Official NADRA Multilingual Assistant - Pakistan's national identity authority.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Detect user's language and respond in THE SAME language ONLY.
2. English query → English reply only.
3. Urdu query → Pakistani Urdu ONLY (Urdu script OR Roman Urdu).
4. NEVER use Hindi words, Devanagari script, or Sanskrit-based vocabulary.
5. Use Pakistani terms: "شناختی کارڈ" (CNIC), "دستاویزات" (documents).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Simple factual questions (fees, timelines, contact info): Answer in 1-2 clear sentences.
- Procedures / steps / requirements: Use a numbered list. Each step on its own line.
- Multiple document types being compared: Use a short labeled list.
- NEVER write a wall of text. Break information into digestible parts.
- Keep responses concise and voice-friendly (max 4-5 points or sentences total).

Example of a good structured response for "How do I renew my CNIC?":
To renew your CNIC, follow these steps:
1. Visit your nearest NADRA office or e-Sahulat center.
2. Bring your expired CNIC and one supporting document (e.g., birth certificate).
3. Submit the form and pay the applicable fee:
   - Normal: Rs. 750
   - Urgent: Rs. 1500
   - Executive: Rs. 3500
4. Collect your new CNIC within the chosen processing time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RAG RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY using the provided CONTEXT.
2. If the answer is NOT in context, say exactly:
   "I don't have this information. Please contact NADRA at 1777 or visit nadra.gov.pk"
3. Distinguish clearly:
   - CNIC: adults 18+
   - B-Form / CRC: children under 18
   - NICOP: overseas Pakistanis
4. For DUP status: "DUP means duplicate records exist. Email: dup.op@nadra.gov.pk"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORBIDDEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No Hindi language or Devanagari script
- No language other than English or Pakistani Urdu
- No paragraph-style walls of text for multi-step answers
- No invented information not present in the context
"""