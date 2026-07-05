# NADRA Multilingual RAG Assistant

A bilingual (English / Pakistani Urdu) conversational assistant for NADRA-related queries (CNIC, B-Form/CRC, NICOP, DUP status, fees, procedures, etc.), built as a **Retrieval-Augmented Generation (RAG)** system with a talking-avatar front end.

The assistant retrieves answers strictly from a curated NADRA knowledge base (ChromaDB), answers in the same language the user asked in, supports voice input/output, and renders a live avatar (via Simli) that speaks the response.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Full Stack Locally](#running-the-full-stack-locally)
- [API Reference](#api-reference)
- [Docker / Deployment (Hugging Face Spaces)](#docker--deployment-hugging-face-spaces)
- [Configuration Reference](#configuration-reference)
- [Known Issues & Troubleshooting](#known-issues--troubleshooting)
- [Security Notes](#security-notes)

---

## Features

- **Bilingual RAG**: Answers are grounded only in retrieved NADRA context — separate Chroma collections for English (`nadra_en`) and Urdu (`nadra_ur`).
- **Automatic language detection**: Detects English vs. Urdu from text (script + `langdetect`), and from speech (Whisper's own language ID).
- **Cross-lingual voice input**: If a user speaks Hindi, Arabic, Farsi, or Punjabi, the transcript is automatically normalized into Pakistani Urdu via an LLM pass before being answered.
- **Speech-to-text**: `faster-whisper` (large-v3) transcribes uploaded audio.
- **Text-to-speech**: `gTTS` + `pydub` convert the answer into 16kHz mono PCM16 audio for the avatar.
- **Talking avatar**: Integrates with the [Simli](https://simli.ai) Avatar API over WebRTC — the answer's audio drives a live lip-synced video avatar in the browser.
- **Graceful degradation**: If Simli is unreachable, the app falls back to text + local audio playback instead of failing.
- **Reranked retrieval**: Top-K candidates from ChromaDB are reranked with a cross-encoder before being passed to the LLM, improving answer relevance.
- **Conversation memory**: The last few turns of chat history are included in both query enhancement and the LLM prompt for contextual follow-ups.
- **React frontend**: A chat UI with mic input, language toggle, and an embedded avatar video panel.

---

## Architecture

```
┌─────────────────────┐        HTTPS / WebRTC        ┌──────────────────────────┐
│   React Frontend     │ ────────────────────────────▶│   FastAPI Backend        │
│   (Vite, frontend/)  │◀──────────────────────────── │   (main.py)              │
└─────────────────────┘         JSON / audio           └────────────┬─────────────┘
                                                                     │
                     ┌───────────────────────────────────────────────┼─────────────────────────────┐
                     ▼                                               ▼                             ▼
          ┌─────────────────────┐                       ┌─────────────────────────┐     ┌─────────────────────┐
          │  faster-whisper      │                       │  Groq LLM (llama-3.3)    │     │  Simli Avatar API    │
          │  (speech → text)     │                       │  - query enhancement      │     │  (WebRTC token +      │
          └─────────────────────┘                       │  - answer generation       │     │   lip-synced video)  │
                                                          │  - Hindi/Arabic → Urdu     │     └─────────────────────┘
                                                          └──────────────┬────────────┘
                                                                         │
                                                          ┌──────────────▼────────────┐
                                                          │   ChromaDB (chroma_db_fixed)│
                                                          │   - nadra_en collection      │
                                                          │   - nadra_ur collection      │
                                                          │   Retrieved via:             │
                                                          │   multilingual-e5-small       │
                                                          │   (embed) + MiniLM cross-      │
                                                          │   encoder (rerank)            │
                                                          └───────────────────────────────┘
```

**Request flow for a text/voice message (`/chat`):**

1. Frontend sends `{ text, history, simli_session_token }` to the backend.
2. Backend detects the input language.
3. The query is rewritten/cleaned by Groq (`enhance_query`) using recent chat history.
4. The enhanced query is embedded and matched against the correct-language Chroma collection (`retrieve_and_rerank`), then reranked with a cross-encoder.
5. Groq generates the final answer, constrained to the retrieved context and the system prompt's language/formatting rules.
6. The answer is converted to speech (`generate_tts`) and returned as base64 PCM audio alongside the text.
7. The frontend streams that PCM audio to the already-open Simli WebRTC session so the avatar speaks the answer; if no session is open, the frontend fetches a fresh Simli token and reconnects.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Speech-to-text | faster-whisper (`large-v3`) |
| Text-to-speech | gTTS + pydub |
| Embeddings | `intfloat/multilingual-e5-small` (SentenceTransformers) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | ChromaDB (persistent, local) |
| Language detection | `langdetect` + Unicode script checks + Whisper's own language ID |
| Avatar | Simli Avatar API (WebRTC) |
| Frontend | React 19 + Vite |
| Deployment target | Docker (CUDA base image), built for Hugging Face Spaces |

---

## Repository Structure

```
.
├── main.py                  # FastAPI backend — all routes and RAG/TTS/Simli logic
├── config.py                 # Non-secret configurables (loads secrets from .env)
├── .env                       # Secrets & environment-specific values (not committed)
├── requirements.txt          # Python dependencies
├── Dockerfile                # CUDA-based image for Hugging Face Spaces deployment
├── chroma_db_fixed/           # Pre-built ChromaDB persistent store (nadra_en / nadra_ur)
└── frontend/                  # React + Vite chat/avatar UI
    ├── src/
    │   ├── App.jsx            # Main chat UI, Simli WebRTC client, mic recorder
    │   ├── api.js              # Backend endpoint URLs (reads VITE_API_URL)
    │   ├── main.jsx             # React entry point
    │   ├── App.css / index.css
    │   └── assets/
    ├── public/
    ├── dist/                    # Production build output (generated by `vite build`)
    ├── package.json
    └── vite.config.js
```

> Note: a duplicate `App.jsx` also exists at the repository root. It is an older, non-responsive version of `frontend/src/App.jsx` (missing the mobile-layout logic). Treat `frontend/src/App.jsx` as the source of truth and remove/ignore the root copy to avoid confusion.

---

## Prerequisites

- **Python 3.10+** (Dockerfile uses 3.11)
- **Node.js 18+** and npm (for the frontend)
- **ffmpeg** and **libsndfile1** installed on the system (required by `pydub`/audio processing)
- A **Groq API key** — https://console.groq.com
- A **Simli API key** and **Face ID** — https://simli.ai
- (Optional but recommended) an **NVIDIA GPU + CUDA** for faster Whisper/embedding inference — the app falls back to CPU automatically if no GPU is available
- A pre-built `chroma_db_fixed/` directory containing the `nadra_en` and `nadra_ur` collections (this repo ships with one already populated)

---

## Environment Variables

All secrets and environment-specific values live in `.env` at the project root (never commit this file). `config.py` loads them via `python-dotenv` and falls back silently to the defaults below if a variable is missing.

| Variable | Required | Default (fallback) | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | *(none)* | API key for Groq (used for the LLM, query enhancement, and Hindi/Arabic→Urdu normalization) |
| `SIMLI_API_KEY` | ✅ | *(none)* | API key for the Simli Avatar API |
| `SIMLI_FACE_ID` | ✅ | `0c2b8b04-5274-41f1-a21c-d5c98322efa9` | The Simli avatar face to render |
| `SIMLI_TOKEN_URL` | Optional | `https://api.simli.ai/compose/token` | Simli endpoint used to mint a WebRTC session token |
| `SIMLI_API_VERSION` | Optional | `v2` | Simli API version sent when requesting a session token |

Example `.env`:

```env
GROQ_API_KEY=your_groq_key_here
SIMLI_API_KEY=your_simli_key_here
SIMLI_FACE_ID=0c2b8b04-5274-41f1-a21c-d5c98322efa9
SIMLI_TOKEN_URL=https://api.simli.ai/compose/token
SIMLI_API_VERSION=v2
```

Everything else (model names, ChromaDB path, retrieval params, voices, the system prompt) is a **non-secret configurable** and lives in `config.py` — see [Configuration Reference](#configuration-reference).

For the frontend, create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

---

## Backend Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install system dependencies (Debian/Ubuntu example)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create your .env file (see above) in the project root

# 5. Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On first run, the backend will:
- Download/load the embedding model, cross-encoder, and Whisper model (cached locally after the first run).
- Connect to the ChromaDB store at `config.CHROMA_PERSIST_DIR` (`/app/chroma_db_fixed` by default) and load the `nadra_en` / `nadra_ur` collections. **These collections must already exist** — this backend does not build the vector store; it only queries it.

Health check:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev          # starts Vite dev server (default: http://localhost:5173)
```

Other scripts:

```bash
npm run build        # production build → frontend/dist
npm run preview      # preview the production build locally
npm run lint          # oxlint
```

Make sure `frontend/.env` points `VITE_API_URL` at your running backend.

---

## Running the Full Stack Locally

1. Start the backend: `uvicorn main:app --reload` (port 8000).
2. Start the frontend: `cd frontend && npm run dev` (port 5173).
3. Open the frontend URL in your browser, grant microphone permission if you want to test voice input, and start chatting.
4. The avatar panel will attempt to connect to Simli automatically on load; if Simli is unreachable, the status indicator will show an error and the app continues to work in text/audio-only mode.

> See [Known Issues](#known-issues--troubleshooting) — the frontend currently calls a slightly different session endpoint than the backend exposes; fix that before expecting the avatar session bootstrap to work out of the box.

---

## API Reference

Base URL: `http://localhost:8000` (or wherever the backend is deployed).

### `GET /`
Serves `index.html` from the working directory (legacy/simple UI hook — the React frontend in `frontend/` is served separately and does not use this route).

### `POST /start-session`
Requests a fresh Simli WebRTC session token. Always returns `200`; check `simli_available` to know if the avatar is usable.

**Response**
```json
{
  "simli_session_token": "abc123...",
  "simli_available": true
}
```

### `POST /chat`
Runs the full RAG pipeline: language detection → query enhancement → retrieval + rerank → Groq answer generation → TTS → Simli token handling.

**Request**
```json
{
  "text": "How do I renew my CNIC?",
  "history": [["previous question", "previous answer"]],
  "simli_session_token": "abc123..."
}
```

**Response**
```json
{
  "answer": "To renew your CNIC, follow these steps: ...",
  "simli_session_token": "abc123...",
  "detected_lang": "en",
  "audio_b64": "base64-encoded-PCM16-mono-16kHz-audio",
  "simli_available": true
}
```

### `POST /transcribe`
Accepts an uploaded audio file (multipart form, field name `audio`) and returns its transcript.

- Uses Whisper's own language detection.
- If the detected language is Hindi/Arabic/Farsi/Punjabi, the text is automatically converted to Pakistani Urdu via Groq before being returned.
- If the language is unsupported (not English/Urdu/convertible), returns a fallback message with `"unsupported": true` and skips further processing.

**Response**
```json
{
  "text": "شناختی کارڈ کیسے بنے گا؟",
  "lang": "ur",
  "unsupported": false
}
```

### `GET /health`
Simple liveness check.

```json
{ "status": "ok" }
```

---

## Docker / Deployment (Hugging Face Spaces)

The included `Dockerfile` is built for Hugging Face Spaces (non-root UID 1000, port `7860`, CUDA runtime base image).

```bash
docker build -t nadra-bot .
docker run --gpus all -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e SIMLI_API_KEY=your_key \
  -e SIMLI_FACE_ID=your_face_id \
  nadra-bot
```

Notes on the current Dockerfile:
- It sets `GROQ_API_KEY` and `SIMLI_FACE_ID` as build-time `ENV` defaults — for a real deployment, prefer passing secrets at **runtime** (`docker run -e ...` / the platform's "Secrets" settings) rather than baking them into the image.
- Model caches (`HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`) are pointed at `/tmp` so they don't require a writable app directory, at the cost of being cleared on every restart.
- ⚠️ It currently does **not** `COPY` `config.py` into the image — see [Known Issues](#known-issues--troubleshooting), this must be fixed before the container will actually start.
- The `chroma_db_fixed/` directory is copied in at build time; for large vector stores, use Git LFS as the comment in the Dockerfile suggests.

---

## Configuration Reference

All of the following live in `config.py` and can be changed without touching `main.py`:

| Setting | Default | Purpose |
|---|---|---|
| `APP_TITLE` | `"NADRA Simli Assistant"` | FastAPI app title |
| `SIMLI_MAX_SESSION_LENGTH` | `3600` | Max Simli session length (seconds) |
| `SIMLI_MAX_IDLE_TIME` | `300` | Max Simli idle time before session ends (seconds) |
| `SIMLI_AUDIO_INPUT_FORMAT` | `"pcm16"` | Audio format sent to Simli |
| `SIMLI_REQUEST_TIMEOUT` | `15` | Timeout for Simli token requests (seconds) |
| `CHROMA_PERSIST_DIR` | `"/app/chroma_db_fixed"` | Path to the ChromaDB persistent store |
| `CHROMA_COLLECTIONS` | `{"en": "nadra_en", "ur": "nadra_ur"}` | Collection names per language |
| `GROQ_MODEL` | `"llama-3.3-70b-versatile"` | Groq model used for all LLM calls |
| `EMBED_MODEL_NAME` | `"intfloat/multilingual-e5-small"` | Sentence embedding model |
| `RERANK_MODEL_NAME` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder reranking model |
| `WHISPER_MODEL_SIZE` | `"large-v3"` | faster-whisper model size |
| `VOICE_EN` / `VOICE_UR` | Neural voice names | Reserved for edge-tts-based voices (currently unused — TTS runs through gTTS; see Known Issues) |
| `TOP_K` | `15` | Candidates fetched from ChromaDB before reranking |
| `FINAL_K` | `5` | Candidates kept after reranking, passed to the LLM |
| `CONVERT_TO_UR_LANGS` | `{"hi", "ar", "fa", "pa"}` | Whisper-detected languages that get auto-converted to Urdu |
| `SYSTEM_PROMPT` | (long NADRA-specific prompt) | Governs language rules, response formatting, and RAG constraints |

---

## Known Issues & Troubleshooting

- **Session endpoint mismatch**: The frontend (`frontend/src/api.js`) calls `POST /start_session` (underscore), but the backend exposes `POST /start-session` (hyphen). This will 404 as-is — align one side to match the other before relying on the avatar auto-connect flow.
- **`config.py` not copied in Docker**: The `Dockerfile` copies `requirements.txt`, `main.py`, and `chroma_db_fixed/`, but not `config.py`. Since `main.py` does `import config`, the container will fail to start until this `COPY` line is added.
- **No `.env` handling in Docker**: The Dockerfile sets some values via `ENV` but doesn't account for `.env`/`python-dotenv` at all — pass secrets via `-e` flags or your platform's secrets manager at runtime instead.
- **`VOICE_EN`/`VOICE_UR` are currently unused**: `edge_tts` is imported but the actual TTS path uses `gTTS` with a plain `"en"`/`"ur"` language code — these two config values aren't wired into `generate_tts()` yet.
- **Duplicate root `App.jsx`**: An older, non-mobile-responsive copy of the frontend lives at the repo root. Safe to delete; `frontend/src/App.jsx` is current.
- **GPU vs CPU**: `device`/`compute_type` are auto-detected (`cuda`/`float16` vs `cpu`/`int8`). Expect noticeably slower Whisper/embedding/reranking performance without a GPU.
- **ChromaDB must pre-exist**: There's no ingestion/build script in this repo — `chroma_db_fixed/` needs to already contain populated `nadra_en`/`nadra_ur` collections, or `/chat` will silently return empty context and fall back to the "I don't have this information" response.

---

## Security Notes

- Never commit `.env`. Add it to `.gitignore` if it isn't already.
- Rotate the Groq and Simli API keys if they were ever hardcoded/committed in an earlier version of this project.
- `CORSMiddleware` is currently configured with `allow_origins=["*"]` — tighten this to your actual frontend origin(s) before deploying publicly.
