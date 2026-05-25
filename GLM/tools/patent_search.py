# drug_discovery_pipeline/tools/patent_search.py
"""
Patent search tool.
Tries PatentsView API (new endpoint), then falls back to a simple
Google Patents HTML scrape, then returns empty if both fail.
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

    # ── Strategy 1: PatentsView API (v3 endpoint) ───────────────────────
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
        # Try the primary endpoint
        resp = requests.post(PATENTSVIEW_BASE, json=payload, timeout=30)
        if resp.status_code >= 400:
            # Try alternate v3 endpoint
            alt_url = PATENTSVIEW_BASE.replace(
                "api.patentsview.org/patents/query",
                "search.patentsview.org/api/v1/patent",
            )
            resp = requests.post(alt_url, json=payload, timeout=30)
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
        logger.warning("PatentsView API failed: %s – trying fallback.", exc)

    # ── Strategy 2: Google Patents scrape (lightweight) ─────────────────
    if not patents:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                )
            }
            params = {
                "q": query,
                "oq": query,
            }
            resp = requests.get(
                "https://patents.google.com/",
                params=params,
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Google Patents search results have articles with patent info
            items = soup.select("article.search-result")[:top_k]
            if not items:
                # Fallback selector
                items = soup.select("[data-result]")[:top_k]

            for item in items:
                title_el = item.select_one("h3, .title, [itemprop='title']")
                num_el = item.select_one("[itemprop='publicationNumber'], .patent-number")
                abstract_el = item.select_one(".abstract, [itemprop='abstract']")
                date_el = item.select_one("[itemprop='publicationDate'], .date")

                title = title_el.get_text(strip=True) if title_el else ""
                num = num_el.get_text(strip=True) if num_el else ""
                abstract = abstract_el.get_text(strip=True) if abstract_el else ""
                date = date_el.get_text(strip=True) if date_el else ""

                if title or num:
                    patents.append({
                        "patent_number": num,
                        "title": title,
                        "abstract": abstract[:500] if abstract else "",
                        "date": date,
                    })
        except Exception as exc:
            logger.warning("Google Patents fallback failed: %s", exc)

    logger.info("Patent search returned %d results for '%s'.", len(patents), query)
    return patents