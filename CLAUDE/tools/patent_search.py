# drug_discovery_pipeline/tools/patent_search.py
# =============================================================================
# FILE: tools/patent_search.py
# ROLE: Patent landscape mining tool.
#       Queries the PatentsView API for drug-related patents.
#       Falls back to a simple WIPO/Google Patents URL scrape if API fails.
#       Extracts chemical compound claims and representative scaffolds.
# =============================================================================

import hashlib
import logging
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import (
    CHROMA_COLLECTION_PATENTS,
    CHROMA_TOP_K,
    PATENTS_MAX_RESULTS,
    PATENTSVIEW_BASE,
)
from utils.helpers import (
    get_or_create_collection,
    query_collection,
    upsert_documents,
)

log = logging.getLogger("drug_discovery.patent_search")


# ---------------------------------------------------------------------------
# PatentsView API
# ---------------------------------------------------------------------------

def _patentsview_search(query: str, max_results: int = PATENTS_MAX_RESULTS) -> list[dict]:
    """
    Query the PatentsView API for patents relevant to *query*.
    Returns a list of patent dicts: {patent_number, title, abstract, date}.

    PatentsView API v1 docs: https://search.patentsview.org/docs/
    """
    endpoint = f"{PATENTSVIEW_BASE}/patent/"
    payload = {
        "q": query,
        "f": ["patent_number", "patent_title", "patent_abstract", "patent_date",
              "assignee_organization", "cpc_subgroup_id"],
        "_per_page": max_results,
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        patents = data.get("patents") or []
        log.info(f"PatentsView returned {len(patents)} patents.")
        return [
            {
                "patent_number": p.get("patent_number", ""),
                "title": p.get("patent_title", ""),
                "abstract": p.get("patent_abstract", ""),
                "date": p.get("patent_date", ""),
                "assignee": p.get("assignee_organization", [{}])[0].get(
                    "assignee_organization", ""
                ) if p.get("assignee_organization") else "",
            }
            for p in patents
            if p.get("patent_abstract")
        ]
    except Exception as e:
        log.warning(f"PatentsView API failed: {e}")
        return []


def _google_patents_scrape(query: str, max_results: int = 5) -> list[dict]:
    """
    Lightweight fallback: scrape Google Patents search result titles.
    NOTE: This is best-effort; Google may block automated requests.
    """
    log.info("Attempting Google Patents scrape fallback...")
    url = f"https://patents.google.com/xhr/query?url=q%3D{requests.utils.quote(query)}&exp=&tags="
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", {}).get("cluster", [])
        patents = []
        for cluster in results[:max_results]:
            for res in cluster.get("result", []):
                patent = res.get("patent", {})
                patents.append(
                    {
                        "patent_number": patent.get("publication_number", ""),
                        "title": patent.get("title", ""),
                        "abstract": patent.get("abstract", ""),
                        "date": patent.get("publication_date", ""),
                        "assignee": patent.get("assignee", ""),
                    }
                )
        log.info(f"Google Patents scrape returned {len(patents)} results.")
        return patents
    except Exception as e:
        log.warning(f"Google Patents scrape failed: {e}")
        return []


# ---------------------------------------------------------------------------
# SMILES / chemical claim extraction
# ---------------------------------------------------------------------------

# Simple regex to find InChI strings or SMILES-like patterns in patent text
_SMILES_PATTERN = re.compile(
    r"(?:^|[\s\(\[])"
    r"([A-Za-z0-9@\[\]\(\)\+\-=#\$%/\\\.]{10,})"
    r"(?:$|[\s\)\]])",
    re.MULTILINE,
)

_INCHI_PATTERN = re.compile(r"InChI=1S?/[A-Za-z0-9/\-\+\(\)\.]+")


def extract_chemical_claims(text: str) -> list[str]:
    """
    Heuristically extract SMILES or InChI strings from patent abstract/claims.
    This is approximate – a real system would use a specialised NLP model.
    """
    smiles_hits = _SMILES_PATTERN.findall(text)
    inchi_hits = _INCHI_PATTERN.findall(text)

    # Filter SMILES candidates: must contain ring notation or heteroatoms
    candidate_smiles = [
        s for s in smiles_hits
        if any(c in s for c in ["c", "n", "o", "C", "N", "O", "F", "Cl"])
        and len(s) > 12
    ]
    return inchi_hits + candidate_smiles[:5]  # cap at 5 per patent


# ---------------------------------------------------------------------------
# Vector store integration
# ---------------------------------------------------------------------------

def _make_patent_id(number: str) -> str:
    h = hashlib.md5(number.encode()).hexdigest()[:8]
    return f"patent_{h}"


def index_patents(patents: list[dict]) -> None:
    """Embed and upsert patents into ChromaDB patents collection."""
    collection = get_or_create_collection(CHROMA_COLLECTION_PATENTS)
    ids, texts, metas = [], [], []
    for p in patents:
        text = f"{p['title']}\n\n{p['abstract']}"
        if not text.strip():
            continue
        doc_id = _make_patent_id(p.get("patent_number") or text[:20])
        ids.append(doc_id)
        texts.append(text)
        metas.append(
            {
                "patent_number": p.get("patent_number", ""),
                "title": p.get("title", "")[:200],
                "date": p.get("date", ""),
                "assignee": p.get("assignee", "")[:100],
            }
        )
    upsert_documents(collection, ids, texts, metas)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def search_patents(
    query: str,
    max_results: int = PATENTS_MAX_RESULTS,
    top_k: int = CHROMA_TOP_K,
    force_refresh: bool = False,
) -> dict:
    """
    Full patent search pipeline:
    1. Query PatentsView (with Google Patents fallback).
    2. Index into ChromaDB.
    3. Semantic search for most relevant results.

    Returns a dict:
      {
        "results": list of ChromaDB result dicts,
        "chemical_claims": list of extracted SMILES/InChI,
        "patent_count": int,
      }
    """
    collection = get_or_create_collection(CHROMA_COLLECTION_PATENTS)

    all_patents = []
    if collection.count() == 0 or force_refresh:
        all_patents = _patentsview_search(query, max_results)
        if not all_patents:
            all_patents = _google_patents_scrape(query)
        if all_patents:
            index_patents(all_patents)
        else:
            log.warning("No patents retrieved from any source.")

    results = query_collection(collection, query, top_k=top_k)

    # Extract chemical claims from all patent texts
    all_text = " ".join([r["document"] for r in results])
    chemical_claims = extract_chemical_claims(all_text)

    return {
        "results": results,
        "chemical_claims": chemical_claims,
        "patent_count": collection.count(),
    }