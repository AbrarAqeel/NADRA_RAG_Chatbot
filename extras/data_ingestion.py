"""
data_ingestion.py
------------------

This script:
1. Loads the final NADRA multilingual JSON (EN / UR / SD).
2. Embeds text for ALL languages (English, Urdu, Sindhi).
3. Uses 'intfloat/multilingual-e5-small' (support for 50+ languages).
4. Creates separate Chroma collections for each domain & language.
5. Prefixes text with "passage: " (required for E5 models).

"""

# ============================================================
#                     CONFIGURATION
# ============================================================

JSON_PATH = r"data\JSON\NADRA_EN_UR_SD.json"
CHROMA_DIR = "chroma_db"

# Multilingual model that supports Urdu, Sindhi, and English well
EMBED_MODEL = "intfloat/multilingual-e5-small"

# E5 models require a prefix for documents to distinguish them from queries
DOC_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Which languages to process from the JSON
TARGET_LANGUAGES = ["en", "ur", "sd"]

DOMAIN_GROUP_MAP = {
    "general": "regulations",
    "products": "identity",
    "projects": "deployments",
}

TEST_QUERY = "How do I renew my CNIC?"

# ============================================================
#                     IMPORTS
# ============================================================

import json
import os
from typing import List, Dict, Any

from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
#                     HELPER FUNCTIONS
# ============================================================

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Loads the JSON dataset from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_chroma(chroma_dir: str) -> chromadb.Client:
    """Initializes the ChromaDB persistent client."""
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


def get_collection_name(domain: str, group: str, lang: str) -> str:
    """
    Generates a consistent collection name.
    Example: general__regulations__ur
    """
    return f"{domain}__{group}__{lang}"


# ============================================================
#                     MAIN INGESTION
# ============================================================

def main():
    print("--- NADRA RAG Ingestion (Multilingual) ---\n")

    # 1. Load Data
    print(f"Loading dataset from: {JSON_PATH}")
    data = load_dataset(JSON_PATH)
    print(f"Loaded {len(data)} chunks.")

    # 2. Load Model
    print(f"\nLoading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    # 3. Init Database
    print(f"Initializing ChromaDB at: {CHROMA_DIR}")
    client = init_chroma(CHROMA_DIR)

    # Cache collections in memory so we don't call get_or_create every loop
    collections_cache: Dict[str, Any] = {}

    print(f"\nStarting ingestion for languages: {TARGET_LANGUAGES}...\n")

    # 4. Main Loop
    for item in tqdm(data, desc="Ingesting", unit="chunk"):

        # Extract routing info
        domain = item["router"]["domain"]
        group = DOMAIN_GROUP_MAP.get(domain, "misc")
        chunk_id = item["chunk_id"]

        # Process each requested language
        for lang in TARGET_LANGUAGES:

            # Safety check: does this language exist in the chunk?
            if lang not in item["content"]:
                continue

            content_obj = item["content"][lang]
            raw_text = content_obj.get("text")

            # Skip if text is missing or empty
            if not raw_text or not raw_text.strip():
                continue

            # Resolve collection for this specific language
            coll_name = get_collection_name(domain, group, lang)

            if coll_name not in collections_cache:
                collections_cache[coll_name] = client.get_or_create_collection(
                    name=coll_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "domain": domain,
                        "group": group,
                        "language": lang,
                        "model": EMBED_MODEL
                    }
                )

            collection = collections_cache[coll_name]

            # IMPORTANT: E5 models need "passage: " prefix for docs
            text_to_embed = f"{DOC_PREFIX}{raw_text}"

            # Generate embedding
            embedding = model.encode(
                text_to_embed,
                normalize_embeddings=True
            ).tolist()

            # Metadata for retrieval
            # We store the raw text without prefix so we display clean text to user later
            metadata = {
                "chunk_id": chunk_id,
                "domain": domain,
                "group": group,
                "language": lang,
                "source_file": item["metadata"]["source_file"],
                "version": item["metadata"]["version"]
            }

            # Add to Chroma
            collection.add(
                ids=[f"{chunk_id}_{lang}"],  # Unique ID per language
                documents=[raw_text],  # Store original text (no prefix)
                embeddings=[embedding],
                metadatas=[metadata]
            )

    print("\nPersisting database...")
    # Chroma 0.4+ persists automatically, but good to mark this step visually.

    # ========================================================
    #                   VALIDATION
    # ========================================================

    print("\n--- Validation Stats ---")
    total_embeddings = 0
    for name, coll in collections_cache.items():
        count = coll.count()
        total_embeddings += count
        print(f"Collection [{name}]: {count} embeddings")

    print(f"\nTotal Embeddings: {total_embeddings}")

    # Simple sanity check
    print("\n--- Running Sanity Check (English) ---")

    # Check English collection for general queries
    test_coll_name = get_collection_name("general", "regulations", "en")

    if test_coll_name in collections_cache:
        test_coll = collections_cache[test_coll_name]

        # Remember: Queries need "query: " prefix for E5
        query_text = f"{QUERY_PREFIX}{TEST_QUERY}"
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]

        results = test_coll.query(
            query_embeddings=[query_emb],
            n_results=1
        )

        if results["documents"]:
            print(f"Query: {TEST_QUERY}")
            print(f"Retrieved: {results['documents'][0][0][:100]}...")
        else:
            print("Warning: No results found for test query.")
    else:
        print(f"Skipping test: Collection {test_coll_name} not created.")

    print("\n=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    main()
