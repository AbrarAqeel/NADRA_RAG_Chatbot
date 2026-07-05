# ---------------------------
# Imports
# ---------------------------

import json
import datetime
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# Configuration
# ---------------------------

INPUT_FILE = r"data\JSON\NADRA_FINAL_DATA_ENGLISH.json"
OUTPUT_FILE = r"data\JSON\NADRA_EN_UR_SD.json"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_LANG = "eng_Latn"
UR_LANG = "urd_Arab"
SD_LANG = "snd_Arab"

MAX_CHARS_PER_SEGMENT = 800
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Model loading
# ---------------------------

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model


# ---------------------------
# Text splitting & joining
# ---------------------------

def split_text(text, max_chars=MAX_CHARS_PER_SEGMENT):
    """
    Splits text into sentence-like chunks without losing order.
    """
    sentences = re.split(r'(?<=[.;])\s+|\n+', text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += (" " if current else "") + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks


def join_text(chunks):
    """
    Re-joins translated chunks into a single string.
    """
    return " ".join(chunks).strip()


# ---------------------------
# Translation helpers
# ---------------------------

def translate_chunks(tokenizer, model, chunks, tgt_lang):
    """
    Translates a list of text chunks to target language.
    """
    results = []

    tokenizer.src_lang = SRC_LANG

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch_chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=512
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        results.extend(decoded)

    return results


def translate_text(tokenizer, model, text, tgt_lang):
    """
    Full pipeline: split → translate → rejoin with sanity checks.
    """
    if not text or not text.strip():
        return None

    chunks = split_text(text)
    translated_chunks = translate_chunks(tokenizer, model, chunks, tgt_lang)
    final_text = join_text(translated_chunks)

    # ---- Quality guardrail ----
    if len(final_text) < 0.4 * len(text):
        return None

    return final_text


# ---------------------------
# Main logic
# ---------------------------

def main():
    print("Loading dataset...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading NLLB-200 model...")
    tokenizer, model = load_model()

    print(f"Translating {len(data)} chunks...\n")

    for item in tqdm(data, desc="Translating"):
        en_text = item["content"]["en"]["text"]

        # ---- Urdu ----
        ur_text = translate_text(
            tokenizer,
            model,
            en_text,
            UR_LANG
        )

        if ur_text:
            item["content"]["ur"]["text"] = ur_text
            item["content"]["ur"]["tokens"] = len(ur_text.split())
            item["content"]["ur"]["status"] = "machine_translated"
        else:
            item["content"]["ur"]["status"] = "translation_failed"

        # ---- Sindhi ----
        sd_text = translate_text(
            tokenizer,
            model,
            en_text,
            SD_LANG
        )

        if sd_text:
            item["content"]["sd"]["text"] = sd_text
            item["content"]["sd"]["tokens"] = len(sd_text.split())
            item["content"]["sd"]["status"] = "machine_translated"
        else:
            item["content"]["sd"]["status"] = "translation_failed"

        # ---- Metadata ----
        item["metadata"]["translation"] = {
            "provider": MODEL_NAME,
            "date": datetime.datetime.utcnow().isoformat(),
            "method": "nllb200_sentence_split"
        }

    print("\nSaving translated dataset...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ Translation complete.")


if __name__ == "__main__":
    main()
