# drug_discovery_pipeline/utils/helpers.py
# =============================================================================
# FILE: utils/helpers.py
# ROLE: Shared utilities — now with DUAL-SERVER aware LLM routing.
#
# Key additions vs v1:
#   • get_ollama_client(server_url)  — parameterised by server
#   • routed_llm_call(task, messages) — looks up TASK_MODEL_MAP, falls back
#   • server_is_healthy(url)         — lightweight liveness probe
#   • llm_call() preserved for backward compat (uses "default" task)
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
import requests

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_SERVER,
    LLM_MAX_TOKENS,
    LLM_RETRIES,
    LLM_RETRY_DELAY,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LOCAL_FALLBACK_MODEL,
    LOCAL_OLLAMA_URL,
    LOG_FILE,
    LOG_LEVEL,
    OLLAMA_API_KEY,
    REMOTE_OLLAMA_URL,
    SERVER_HEALTH_TIMEOUT,
    TASK_MODEL_MAP,
)


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(name: str = "drug_discovery") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning(f"Cannot open log file {LOG_FILE}: {e}")

    return logger


log = setup_logging()


# ===========================================================================
# Server health check
# ===========================================================================

# In-process cache: {url -> (is_healthy: bool, checked_at: float)}
_health_cache: dict[str, tuple[bool, float]] = {}
_HEALTH_CACHE_TTL = 60.0  # re-check every 60 s


def server_is_healthy(base_url: str) -> bool:
    """
    Probe the Ollama server's /api/tags endpoint.
    Results are cached for _HEALTH_CACHE_TTL seconds to avoid hammering.
    """
    now = time.time()
    cached = _health_cache.get(base_url)
    if cached and (now - cached[1]) < _HEALTH_CACHE_TTL:
        return cached[0]

    # Strip /v1 suffix to get the base ollama URL
    probe_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        resp = requests.get(probe_url, timeout=SERVER_HEALTH_TIMEOUT)
        healthy = resp.status_code == 200
    except Exception:
        healthy = False

    _health_cache[base_url] = (healthy, now)
    status = "✓ healthy" if healthy else "✗ unreachable"
    log.debug(f"Server health {base_url}: {status}")
    return healthy


def invalidate_health_cache(base_url: str) -> None:
    """Force a re-check on next call."""
    _health_cache.pop(base_url, None)


# ===========================================================================
# Ollama / OpenAI clients
# ===========================================================================

# Client cache (one per server URL)
_clients: dict[str, openai.OpenAI] = {}


def get_ollama_client(server_url: str = REMOTE_OLLAMA_URL) -> openai.OpenAI:
    """
    Return a cached OpenAI-compatible client for *server_url*.
    Creates a new client on first call for each URL.
    """
    if server_url not in _clients:
        _clients[server_url] = openai.OpenAI(
            base_url=server_url,
            api_key=OLLAMA_API_KEY,
            timeout=LLM_TIMEOUT,
        )
    return _clients[server_url]


# ===========================================================================
# Core LLM call with routing
# ===========================================================================

def routed_llm_call(
    task: str,
    messages: list[dict],
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    retries: int = LLM_RETRIES,
    retry_delay: float = LLM_RETRY_DELAY,
) -> str:
    """
    Route an LLM call to the correct server+model based on *task*.

    Routing logic:
    1. Look up (model, server) from TASK_MODEL_MAP.
    2. If that server is unhealthy, fall back to the other server with the
       same model (if it has it) or to the local fallback model.
    3. Retry up to *retries* times on transient errors.

    Args:
        task: Key into TASK_MODEL_MAP (e.g. "hypothesis", "retrosynthesis").
        messages: OpenAI-format message list.

    Returns:
        LLM response text.
    """
    model, server = TASK_MODEL_MAP.get(task, TASK_MODEL_MAP["default"])

    # ── Health-aware failover ────────────────────────────────────────────
    primary_ok = server_is_healthy(server)
    if not primary_ok:
        alt_server = LOCAL_OLLAMA_URL if server == REMOTE_OLLAMA_URL else REMOTE_OLLAMA_URL
        alt_ok = server_is_healthy(alt_server)
        if alt_ok:
            log.warning(
                f"Primary server {server} is down for task '{task}'. "
                f"Falling back to {alt_server} with model {LOCAL_FALLBACK_MODEL}."
            )
            server = alt_server
            # Use fallback model only if switching to local
            if alt_server == LOCAL_OLLAMA_URL:
                model = LOCAL_FALLBACK_MODEL
        else:
            log.error("Both servers appear unreachable. Attempting anyway...")

    log.info(f"[{task}] → {model} @ {server}")
    client = get_ollama_client(server)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            log.debug(f"[{task}] response: {len(content)} chars")
            return content

        except openai.APIConnectionError as e:
            log.warning(f"[{task}] Connection error attempt {attempt}: {e}")
            invalidate_health_cache(server)
            last_exc = e
        except openai.APIStatusError as e:
            log.warning(f"[{task}] API status {e.status_code} attempt {attempt}: {e.message}")
            last_exc = e
        except Exception as e:
            log.warning(f"[{task}] Unexpected error attempt {attempt}: {e}")
            last_exc = e

        if attempt < retries:
            time.sleep(retry_delay)

    log.error(f"[{task}] All {retries} attempts failed.")
    raise last_exc or RuntimeError(f"LLM call failed for task '{task}'")


def llm_call(
    messages: list[dict],
    model: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    retries: int = LLM_RETRIES,
    retry_delay: float = LLM_RETRY_DELAY,
) -> str:
    """
    Backward-compatible generic LLM call.
    If *model* is specified it tries to infer which server has that model.
    Otherwise routes via the "default" task (REMOTE gemma4:31b).
    """
    if model is None:
        return routed_llm_call(
            "default", messages, temperature, max_tokens, retries, retry_delay
        )

    # Model specified explicitly — determine server
    if model in (
        "gemma4:31b-it-q8_0", "gemma4:26b-a4b-it-q8_0",
        "gemma4:e4b-it-bf16", "gpt-oss:20b",
    ):
        server = REMOTE_OLLAMA_URL
    else:
        server = LOCAL_OLLAMA_URL

    log.info(f"[explicit] → {model} @ {server}")
    client = get_ollama_client(server)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"Explicit LLM call attempt {attempt} failed: {e}")
            last_exc = e
            if attempt < retries:
                time.sleep(retry_delay)

    raise last_exc or RuntimeError("LLM call failed")


# ===========================================================================
# Embeddings  (prefer REMOTE for throughput; fall back to LOCAL)
# ===========================================================================

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Embed *text*. Prefers REMOTE server; falls back to LOCAL.
    """
    for server in (EMBEDDING_SERVER, LOCAL_OLLAMA_URL):
        if not server_is_healthy(server):
            continue
        try:
            client = get_ollama_client(server)
            resp = client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except Exception as e:
            log.warning(f"Embedding failed on {server}: {e}")
            invalidate_health_cache(server)

    log.error("Embedding failed on all servers.")
    return []


def get_embeddings_batch(
    texts: list[str], model: str = EMBEDDING_MODEL
) -> list[list[float]]:
    """Embed a list of texts, returning a list of vectors."""
    return [get_embedding(t, model) for t in texts]


# ===========================================================================
# ChromaDB
# ===========================================================================

_chroma_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        log.info(f"ChromaDB initialised at {CHROMA_DIR}")
    return _chroma_client


def get_or_create_collection(name: str) -> chromadb.Collection:
    client = get_chroma_client()
    col = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    log.debug(f"Collection '{name}' has {col.count()} docs")
    return col


def upsert_documents(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    if not texts:
        return
    log.info(f"Embedding {len(texts)} docs → '{collection.name}'...")
    embeddings = get_embeddings_batch(texts)
    valid = [
        (i, t, e, m)
        for i, t, e, m in zip(
            ids, texts, embeddings,
            metadatas or [{}] * len(texts)
        )
        if e
    ]
    if not valid:
        log.warning("No valid embeddings; skipping upsert.")
        return
    v_ids, v_docs, v_embs, v_metas = zip(*valid)
    collection.upsert(
        ids=list(v_ids), documents=list(v_docs),
        embeddings=list(v_embs), metadatas=list(v_metas),
    )
    log.info(f"Upserted {len(v_ids)} documents.")


def query_collection(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    q_emb = get_embedding(query)
    if not q_emb:
        return []
    n = min(top_k, max(collection.count(), 1))
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i in range(len(results["ids"][0])):
        out.append({
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return out


# ===========================================================================
# JSON extraction
# ===========================================================================

def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            return json.loads(brace.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON from:\n{text[:400]}")


def safe_extract_json(text: str, default: Any = None) -> Any:
    try:
        return extract_json(text)
    except (ValueError, TypeError) as e:
        log.warning(f"JSON extraction failed: {e}")
        return default


def repair_json_with_llm(broken_text: str) -> Any:
    """
    Use the local deepseek-coder model to repair malformed JSON.
    Only invoked when safe_extract_json returns None.
    """
    from utils.prompts import json_repair_prompt
    try:
        messages = json_repair_prompt(broken_text)
        fixed = routed_llm_call("json_repair", messages, temperature=0.0)
        return extract_json(fixed)
    except Exception as e:
        log.warning(f"JSON repair failed: {e}")
        return default if False else None  # keeps linters happy


# ===========================================================================
# File I/O
# ===========================================================================

def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info(f"Saved: {path}")


def load_text(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    log.warning(f"File not found: {path}")
    return None


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def run_id() -> str:
    return timestamp()


def truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [+{len(text)-max_chars} chars]"


def flatten_list(nested: list) -> list:
    return [item for sub in nested for item in (sub if isinstance(sub, list) else [sub])]