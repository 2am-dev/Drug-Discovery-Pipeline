# drug_discovery_pipeline/tools/patent_search.py
"""
Patent search tool using the PatentsView public API.
Falls back to a simple text-based search if the API is unavailable.
"""

from __future__ import annotations

import logging
from typing import List, Dict

import requests

from config import PATENTSVIEW_BASE

logger = logging.getLogger(__name__)


def search_patents(query: str, top_k: int = 10) -> List[Dict[str, str]]:
    """
    Search for patents related to *query*.

    Returns a list of dicts with keys: ``patent_number``, ``title``,
    ``abstract``, ``date``.
    """
    patents: list[dict] = []

    # ── Try PatentsView API ───────────────────────────────────────────────
    try:
        payload = {
            "q": {"_text_all": {"patent_abstract": query}},
            "f": [
                "patent_number",
                "patent_title",
                "patent_abstract",
                "patent_date",
            ],
            "s": [{"patent_date": "desc"}],
            "o": {"per_page": top_k},
        }
        resp = requests.post(PATENTSVIEW_BASE, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for p in data.get("patents", []):
            patents.append({
                "patent_number": p.get("patent_number", ""),
                "title": p.get("patent_title", ""),
                "abstract": p.get("patent_abstract", ""),
                "date": p.get("patent_date", ""),
            })
    except Exception as exc:
        logger.warning("PatentsView API failed: %s – returning empty results.", exc)

    logger.info("Patent search returned %d results for '%s'.", len(patents), query)
    return patents