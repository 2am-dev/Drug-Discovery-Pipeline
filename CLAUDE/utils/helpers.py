# drug_discovery_pipeline/utils/helpers.py
# =============================================================================
# FILE: utils/helpers.py
# ROLE: Shared utility functions used across the whole pipeline.
#       Includes: logging setup, Ollama client factory, ChromaDB initialiser,
#       embedding helper, JSON extraction from LLM responses, file I/O.
# =============================================================================

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
import openai
from chromadb.config import Settings

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LOG_FILE,
    LOG_LEVEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(name: str = "drug_discovery") -> logging.Logger:
    """
    Configure and return a named logger that writes to both stdout and a file.
    Call once from main.py; subsequent getLogger calls return the same instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning(f"Could not open log file {LOG_FILE}: {e}")

    return logger


# Module-level logger
log = setup_logging()


# ===========================================================================
# Ollama / OpenAI client
# ===========================================================================

def get_ollama_client() -> openai.OpenAI:
    """
    Return a configured OpenAI client pointing at the local Ollama server.
    The client is cheap to create; no persistent connection is established.
    """
    return openai.OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=LLM_TIMEOUT,
    )


def llm_call(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Make a chat completion call to Ollama with automatic retry on transient
    failures.  Returns the assistant's text content as a plain string.
    """
    client = get_ollama_client()
    for attempt in range(1, retries + 1):
        try:
            log.debug(f"LLM call attempt {attempt}/{retries} – model={model}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            log.debug(f"LLM response length: {len(content)} chars")
            return content
        except Exception as e:
            log.warning(f"LLM call attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(retry_delay)
            else:
                log.error("All LLM retry attempts exhausted.")
                raise


# ===========================================================================
# Embeddings
# ===========================================================================

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Embed *text* using the Ollama embedding model.
    Returns a list of floats (the embedding vector).
    """
    client = get_ollama_client()
    try:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        log.error(f"Embedding failed for text snippet: {e}")
        return []


def get_embeddings_batch(
    texts: list[str], model: str = EMBEDDING_MODEL
) -> list[list[float]]:
    """Embed a list of texts, returning a list of vectors."""
    return [get_embedding(t, model) for t in texts]


# ===========================================================================
# ChromaDB
# ===========================================================================

_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return (or lazily create) the singleton ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        log.info(f"ChromaDB initialised at {CHROMA_DIR}")
    return _chroma_client


def get_or_create_collection(name: str) -> chromadb.Collection:
    """Get an existing ChromaDB collection or create it if absent."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    log.debug(f"Collection '{name}' has {collection.count()} documents")
    return collection


def upsert_documents(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    metadatas: Optional[list[dict]] = None,
) -> None:
    """
    Embed *texts* and upsert them (with their embeddings) into *collection*.
    Existing documents with the same id are overwritten.
    """
    if not texts:
        return
    log.info(f"Embedding {len(texts)} documents for collection '{collection.name}'...")
    embeddings = get_embeddings_batch(texts)
    valid = [
        (i, t, e, m)
        for i, t, e, m in zip(
            ids, texts, embeddings, metadatas or [{}] * len(texts)
        )
        if e  # skip failed embeddings
    ]
    if not valid:
        log.warning("No valid embeddings produced; skipping upsert.")
        return
    v_ids, v_docs, v_embs, v_metas = zip(*valid)
    collection.upsert(
        ids=list(v_ids),
        documents=list(v_docs),
        embeddings=list(v_embs),
        metadatas=list(v_metas),
    )
    log.info(f"Upserted {len(v_ids)} documents.")


def query_collection(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Embed *query* and return the top-k nearest documents from *collection*.
    Each result is a dict: {id, document, metadata, distance}.
    """
    q_emb = get_embedding(query)
    if not q_emb:
        return []
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i in range(len(results["ids"][0])):
        out.append(
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return out


# ===========================================================================
# JSON extraction from LLM responses
# ===========================================================================

def extract_json(text: str) -> Any:
    """
    Robustly parse a JSON object from LLM output.
    Handles cases where the model wraps JSON in markdown code fences.
    Returns the parsed Python object, or raises ValueError on total failure.
    """
    # 1. Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first { ... } block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


def safe_extract_json(text: str, default: Any = None) -> Any:
    """Like extract_json but returns *default* instead of raising."""
    try:
        return extract_json(text)
    except (ValueError, TypeError) as e:
        log.warning(f"JSON extraction failed: {e}")
        return default


# ===========================================================================
# File I/O helpers
# ===========================================================================

def save_text(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info(f"Saved: {path}")


def load_text(path: Path) -> Optional[str]:
    """Read and return text from *path*, or None if the file does not exist."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    log.warning(f"File not found: {path}")
    return None


def timestamp() -> str:
    """Return a filesystem-safe ISO-8601 timestamp string."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def run_id() -> str:
    """Generate a short unique run identifier."""
    return timestamp()


# ===========================================================================
# Misc
# ===========================================================================

def truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate *text* to *max_chars* with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text)-max_chars} chars]"


def flatten_list(nested: list) -> list:
    """Flatten one level of nesting."""
    return [item for sublist in nested for item in (sublist if isinstance(sublist, list) else [sublist])]