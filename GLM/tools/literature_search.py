# drug_discovery_pipeline/tools/literature_search.py
"""
Literature search tool.
1. Queries a local ChromaDB collection (pre-populated with PubMed abstracts).
2. If ChromaDB is empty, falls back to NCBI Entrez E-Utilities to fetch
   PubMed abstracts, embeds them, and stores them in ChromaDB for future use.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import List, Dict

import chromadb
import requests

from config import (
    ENTREZ_BASE,
    ENTREZ_EMAIL,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)
from utils.helpers import get_embeddings

logger = logging.getLogger(__name__)

# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB literature collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name="literature",
        metadata={"description": "PubMed abstracts for drug-discovery pipeline"},
    )


def _collection_has_data(collection: chromadb.Collection) -> bool:
    return collection.count() > 0


# ── Entrez / PubMed ──────────────────────────────────────────────────────────

def _entrez_search(query: str, retmax: int = 20) -> List[str]:
    """Return a list of PMIDs matching *query* via ESearch."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "email": ENTREZ_EMAIL,
        "usehistory": "n",
    }
    try:
        r = requests.get(f"{ENTREZ_BASE}/esearch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        logger.error("Entrez ESearch failed: %s", exc)
        return []


def _entrez_fetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    """Fetch title + abstract for each PMID via EFetch (XML)."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": ENTREZ_EMAIL,
    }
    try:
        r = requests.get(f"{ENTREZ_BASE}/efetch.fcgi", params=params, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        logger.error("Entrez EFetch failed: %s", exc)
        return []

    articles: list[dict] = []
    try:
        root = ET.fromstring(r.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            pmid = pmid_el.text if pmid_el is not None else ""
            title = title_el.text if title_el is not None else ""
            abstract = abstract_el.text if abstract_el is not None else ""
            if abstract:
                articles.append({
                    "pmid": pmid,
                    "title": title or "(no title)",
                    "abstract": abstract,
                })
    except ET.ParseError as exc:
        logger.error("Failed to parse Entrez XML: %s", exc)
    return articles


# ── Populate ChromaDB ────────────────────────────────────────────────────────

def _populate_chroma(collection: chromadb.Collection, query: str, retmax: int = 25) -> None:
    """Fetch abstracts from PubMed, embed, and store in ChromaDB."""
    logger.info("Populating ChromaDB with PubMed results for query: '%s'", query)
    pmids = _entrez_search(query, retmax=retmax)
    if not pmids:
        logger.warning("No PMIDs found for query: '%s'", query)
        return
    time.sleep(0.4)  # be nice to NCBI
    articles = _entrez_fetch_abstracts(pmids)
    if not articles:
        return

    texts = [f"{a['title']} {a['abstract']}" for a in articles]
    ids = [f"pmid_{a['pmid']}" for a in articles]
    metas = [{"pmid": a["pmid"], "title": a["title"]} for a in articles]

    # Batch embeddings (respect Ollama limits)
    batch_size = 32
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(get_embeddings(batch))

    collection.add(ids=ids, documents=texts, embeddings=all_embeddings, metadatas=metas)
    logger.info("Stored %d articles in ChromaDB.", len(articles))


# ── Public API ────────────────────────────────────────────────────────────────

def search_literature(query: str, top_k: int = 10) -> List[Dict[str, str]]:
    """
    Search for literature relevant to *query*.

    Returns a list of dicts with keys: ``pmid``, ``title``, ``abstract`` (or
    ``text`` when coming from ChromaDB).
    """
    collection = _get_collection()

    if not _collection_has_data(collection):
        logger.info("ChromaDB collection is empty – fetching from PubMed…")
        _populate_chroma(collection, query, retmax=30)

    # Semantic search via ChromaDB
    query_embedding = get_embeddings([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))

    articles: list[dict] = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            articles.append({
                "pmid": meta.get("pmid", "unknown"),
                "title": meta.get("title", ""),
                "text": doc,
            })
    logger.info("Literature search returned %d results.", len(articles))
    return articles