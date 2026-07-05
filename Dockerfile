# ============================================================
# NADRA Backend — Hugging Face Spaces Dockerfile
# HF requirement: non-root user UID 1000, port 7860
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

# ── HF requires a non-root user with UID 1000 ────────────────
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── App files ─────────────────────────────────────────────────
COPY --chown=user main.py .

# ChromaDB is copied in from the repo (uploaded via Git LFS)
COPY --chown=user chroma_db_fixed/ ./chroma_db_fixed/

# ── Environment defaults ──────────────────────────────────────
ENV GROQ_API_KEY=""
ENV SIMLI_FACE_ID="0c2b8b04-5274-41f1-a21c-d5c98322efa9"
ENV CHROMA_PERSIST_DIR="/app/chroma_db_fixed"
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False
ENV CHROMA_TELEMETRY=False

# Cache dirs inside /tmp so models survive across the session
ENV HF_HOME="/tmp/hf_cache"
ENV TRANSFORMERS_CACHE="/tmp/hf_cache"
ENV TORCH_HOME="/tmp/torch_cache"

# ── HF Spaces requires port 7860 ──────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
