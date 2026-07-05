# ---------------------------
# Imports
# ---------------------------

import json
import re
from typing import List


# ---------------------------
# Configuration
# ---------------------------

MAX_WORDS_PER_CHUNK = 300
OVERLAP_RATIO = 0.18

OUTPUT_JSON = r"data\JSON\NADRA_FINAL_DATA_ENGLISH.json"

# ---------------------------
# Routing configuration (FINAL — DO NOT CHANGE)
# ---------------------------

ROUTING_CONFIG = {
    "NADRA_GENERAL.txt": {
        "domain": "general",
        "group": "regulations"
    },
    "NADRA_PRODUCTS.txt": {
        "domain": "products",
        "group": "identity"
    },
    "NADRA_PROJECTS.txt": {
        "domain": "projects",
        "group": "deployments"
    }
}


# ---------------------------
# Utility functions
# ---------------------------

def read_txt(file_path: str) -> str:
    """
    Reads a UTF-8 encoded text file and returns raw text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Splits text into paragraphs.

    Paragraphs are separated by one or more blank lines.
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_paragraphs(
    paragraphs: List[str],
    max_words: int,
    overlap_ratio: float
) -> List[str]:
    """
    Combines paragraphs into semantic chunks with paragraph-based overlap.
    """
    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > max_words:
            chunks.append(" ".join(current_chunk))

            overlap_count = max(1, int(len(current_chunk) * overlap_ratio))
            current_chunk = current_chunk[-overlap_count:]
            current_words = sum(len(p.split()) for p in current_chunk)

        current_chunk.append(para)
        current_words += para_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_chunk_id(
    domain: str,
    group: str,
    index: int
) -> str:
    """
    Generates deterministic, stable chunk IDs.

    Format:
    {domain}_{group}_{index}
    """
    return f"{domain}_{group}_{index:03d}"


def create_chunk_object(
    chunk_id: str,
    domain: str,
    group: str,
    text: str,
    source_file: str,
    overlap_with: List[str]
) -> dict:
    """
    Creates a finalized JSON chunk object.
    """
    return {
        "chunk_id": chunk_id,
        "router": {
            "domain": domain,
            "group": group
        },
        "content": {
            "en": {
                "text": text,
                "tokens": len(text.split()),
                "status": "final"
            },
            "ur": {
                "text": None,
                "tokens": None,
                "status": "pending"
            },
            "sd": {
                "text": None,
                "tokens": None,
                "status": "pending"
            }
        },
        "chunking": {
            "is_overlapping": bool(overlap_with),
            "overlap_with": overlap_with,
            "overlap_ratio": OVERLAP_RATIO if overlap_with else 0.0
        },
        "metadata": {
            "source_file": source_file,
            "language_canon": "en",
            "version": "1.0"
        }
    }


# ---------------------------
# Main processing logic
# ---------------------------

def process_file(file_path: str) -> List[dict]:
    """
    Processes a single TXT file into JSON chunks.
    """
    file_name = file_path.split("/")[-1]
    routing = ROUTING_CONFIG[file_name]

    raw_text = read_txt(file_path)
    paragraphs = split_into_paragraphs(raw_text)
    chunks = chunk_paragraphs(paragraphs, MAX_WORDS_PER_CHUNK, OVERLAP_RATIO)

    json_chunks = []

    for i, chunk_text in enumerate(chunks, start=1):
        chunk_id = generate_chunk_id(
            routing["domain"],
            routing["group"],
            i
        )

        overlap_with = []
        if i > 1:
            overlap_with.append(
                generate_chunk_id(
                    routing["domain"],
                    routing["group"],
                    i - 1
                )
            )

        json_chunks.append(
            create_chunk_object(
                chunk_id=chunk_id,
                domain=routing["domain"],
                group=routing["group"],
                text=chunk_text,
                source_file=file_name,
                overlap_with=overlap_with
            )
        )

    return json_chunks


# ---------------------------
# Entry point
# ---------------------------

def main():
    """
    Builds the final canonical NADRA English dataset.
    """

    files = [
        r"data\TXT\NADRA_GENERAL_CLEANED.txt",
        r"data\TXT\NADRA_PRODUCTS_CLEANED.txt",
        r"data\TXT\NADRA_PROJECTS_CLEANED.txt"
    ]

    all_chunks = []

    for file_path in files:
        all_chunks.extend(process_file(file_path))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_chunks)} chunks to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()