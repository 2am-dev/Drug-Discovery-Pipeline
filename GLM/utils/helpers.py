# drug_discovery_pipeline/utils/helpers.py
"""
Shared utilities – dual-server LLM routing, embeddings, health check.
Handles None content, empty responses, and automatic retry.
"""

from __future__ import annotations

import logging
import time
from typing import List

from openai import OpenAI

import config

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Client cache ──────────────────────────────────────────────────────────────

_clients: dict[str, OpenAI] = {}


def _get_client(base_url: str, api_key: str) -> OpenAI:
    key = f"{base_url}:{api_key}"
    if key not in _clients:
        _clients[key] = OpenAI(base_url=base_url, api_key=api_key, timeout=config.LLM_TIMEOUT)
    return _clients[key]


def _get_remote_client() -> OpenAI:
    return _get_client(config.REMOTE_OLLAMA_BASE_URL, config.REMOTE_OLLAMA_API_KEY)


def _get_local_client() -> OpenAI:
    return _get_client(config.LOCAL_OLLAMA_BASE_URL, config.LOCAL_OLLAMA_API_KEY)


def _get_embedding_client() -> OpenAI:
    return _get_client(config.EMBEDDING_BASE_URL, config.EMBEDDING_API_KEY)


# ── Route resolution ─────────────────────────────────────────────────────────

def _resolve_route(task: str) -> str:
    return config.TASK_ROUTES.get(task, "heavy")


def _get_models_for_tier(tier: str) -> List[str]:
    if tier == "heavy":
        return [config.REMOTE_LLM_MODEL] + list(config.REMOTE_LLM_FALLBACKS)
    return [config.LOCAL_LLM_MODEL] + list(config.LOCAL_LLM_FALLBACKS)


def _model_client_map() -> dict[str, tuple[OpenAI, str]]:
    m: dict[str, tuple[OpenAI, str]] = {}
    for model in [config.REMOTE_LLM_MODEL] + list(config.REMOTE_LLM_FALLBACKS):
        m[model] = (_get_remote_client(), "heavy")
    for model in [config.LOCAL_LLM_MODEL] + list(config.LOCAL_LLM_FALLBACKS):
        m[model] = (_get_local_client(), "light")
    return m


# ── Core LLM call with automatic failover ─────────────────────────────────────

def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    task: str = "heavy",
) -> str:
    """
    Send a chat-completion request, routing to the correct server
    based on *task*.  Falls back through the model list on failure.
    Retries once on empty response.
    """
    logger = logging.getLogger("helpers.call_llm")
    tier = _resolve_route(task)
    models = _get_models_for_tier(tier)
    client = _get_remote_client() if tier == "heavy" else _get_local_client()
    mmap = _model_client_map()

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None

    for model in models:
        current_client, _ = mmap.get(model, (client, tier))

        for attempt in range(1, config.LLM_MAX_RETRIES + 1):
            try:
                logger.debug(
                    "Calling model=%s  tier=%s  task=%s  attempt=%d",
                    model, tier, task, attempt,
                )
                resp = current_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # ── Null-safe content extraction ─────────────────────
                raw_content = resp.choices[0].message.content
                content = (raw_content or "").strip()

                # ── Retry once on empty response ─────────────────────
                if not content and attempt == 1:
                    logger.warning(
                        "Empty response from model=%s task=%s – retrying…",
                        model, task,
                    )
                    # Add a nudge message
                    nudge_messages = messages + [
                        {"role": "assistant", "content": ""},
                        {"role": "user", "content": "Please provide your answer now. Do not leave it blank."},
                    ]
                    resp2 = current_client.chat.completions.create(
                        model=model,
                        messages=nudge_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    raw_content2 = resp2.choices[0].message.content
                    content = (raw_content2 or "").strip()

                if content:
                    logger.info("✓ task=%s  model=%s  tier=%s  chars=%d", task, model, tier, len(content))
                    return content
                else:
                    logger.warning("Still empty after nudge – model=%s task=%s", model, task)

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM call failed  model=%s  attempt=%d/%d: %s",
                    model, attempt, config.LLM_MAX_RETRIES, exc,
                )
                if attempt < config.LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)

        logger.warning("Model %s exhausted all retries – trying next fallback.", model)

    raise RuntimeError(
        f"All LLM models failed for task='{task}'. Last error: {last_exc}"
    )


# ── Convenience wrappers ──────────────────────────────────────────────────────

def call_llm_heavy(prompt: str, **kwargs) -> str:
    return call_llm(prompt, task="heavy", **kwargs)


def call_llm_light(prompt: str, **kwargs) -> str:
    return call_llm(prompt, task="light", **kwargs)


# ── Embeddings (always local) ─────────────────────────────────────────────────

def get_embeddings(texts: List[str]) -> List[List[float]]:
    logger = logging.getLogger("helpers.get_embeddings")
    client = _get_embedding_client()
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    logger.info("Embedded %d texts with %s.", len(texts), config.EMBEDDING_MODEL)
    return [item.embedding for item in resp.data]


# ── Health check ──────────────────────────────────────────────────────────────

def check_servers() -> dict:
    """
    Ping both Ollama servers + embedding endpoint.
    Returns a dict with ok/error per server.
    """
    logger = logging.getLogger("helpers.check_servers")
    results: dict = {}

    for label, base_url, api_key, model in [
        ("remote",     config.REMOTE_OLLAMA_BASE_URL, config.REMOTE_OLLAMA_API_KEY, config.REMOTE_LLM_MODEL),
        ("local",      config.LOCAL_OLLAMA_BASE_URL,  config.LOCAL_OLLAMA_API_KEY,  config.LOCAL_LLM_MODEL),
        ("embeddings", config.EMBEDDING_BASE_URL,     config.EMBEDDING_API_KEY,     config.EMBEDDING_MODEL),
    ]:
        try:
            c = _get_client(base_url, api_key)
            if label == "embeddings":
                c.embeddings.create(model=model, input=["ping"])
                results[label] = {"ok": True, "model": model}
            else:
                resp = c.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                results[label] = {"ok": True, "model": resp.model}
            logger.info("✓ %s server OK – %s @ %s", label, model, base_url)
        except Exception as exc:
            results[label] = {"ok": False, "error": str(exc)}
            logger.error("✗ %s server FAILED – %s @ %s: %s", label, model, base_url, exc)

    return results


# ── Misc helpers ──────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)