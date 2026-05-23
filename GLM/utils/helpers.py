# drug_discovery_pipeline/utils/helpers.py
"""
Shared utilities: LLM / embedding clients, logging setup, common helpers.
"""

from __future__ import annotations

import logging
import time
from typing import List

from openai import OpenAI

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    LOG_LEVEL,
)

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure root logger for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── LLM client singleton ─────────────────────────────────────────────────────

_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """Return a singleton OpenAI client pointed at the local Ollama server."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            timeout=LLM_TIMEOUT,
        )
    return _llm_client


# ── LLM call wrapper ─────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Send a chat-completion request to the Ollama-hosted LLM.
    Retries up to ``LLM_MAX_RETRIES`` times with exponential back-off.
    """
    logger = logging.getLogger("helpers.call_llm")
    client = get_llm_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, LLM_MAX_RETRIES, exc)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts: {exc}") from exc
    return ""  # unreachable but keeps mypy happy


# ── Embedding wrapper ─────────────────────────────────────────────────────────

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Return embeddings for *texts* using the configured Ollama embedding model."""
    logger = logging.getLogger("helpers.get_embeddings")
    client = get_llm_client()
    # Ollama's OpenAI-compatible endpoint handles batched inputs
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ── Misc helpers ──────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a file name component."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)