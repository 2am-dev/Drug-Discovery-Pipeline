# drug_discovery_pipeline/tools/literature_search.py
# =============================================================================
# FILE: tools/literature_search.py
# ROLE: Literature mining tool.
#       1. Queries PubMed via NCBI Entrez REST (efetch / esearch).
#       2. Embeds retrieved abstracts with nomic-embed-text.
#       3. Stores them in ChromaDB for semantic retrieval.
#       4. Returns top-k relevant abstracts for a given query.
# =============================================================================

import hashlib
import logging
import time
from typing import Optional

import requests
import feedparser  # fallback RSS parsing

from config import (
    CHROMA_COLLECTION_LITERATURE,
    CHROMA_TOP_K,
    PUBMED_ENTREZ_BASE,
    PUBMED_MAX_RESULTS,
)
from utils.helpers import (
    get_or_create_collection,
    query_collection,
    upsert_documents,
)

log = logging.getLogger("drug_discovery.literature_search")


# ---------------------------------------------------------------------------
# PubMed / Entrez helpers
# ---------------------------------------------------------------------------

def _entrez_search(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[str]:
    """
    Run an Entrez esearch and return a list of PubMed IDs (PMIDs).
    Falls back to an empty list on network failure.
    """
    url = f"{PUBMED_ENTREZ_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        log.info(f"PubMed esearch returned {len(pmids)} PMIDs for '{query}'")
        return pmids
    except Exception as e:
        log.error(f"PubMed esearch failed: {e}")
        return []


def _entrez_fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch abstracts for a list of PMIDs via efetch (XML mode).
    Returns a list of dicts: {pmid, title, abstract, authors, year}.
    """
    if not pmids:
        return []
    url = f"{PUBMED_ENTREZ_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"PubMed efetch failed: {e}")
        return []

    # Parse XML with BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "xml")
        articles = []
        for article in soup.find_all("PubmedArticle"):
            pmid_tag = article.find("PMID")
            title_tag = article.find("ArticleTitle")
            abstract_tag = article.find("AbstractText")
            year_tag = article.find("PubDate")
            author_tags = article.find_all("LastName")

            pmid = pmid_tag.get_text(strip=True) if pmid_tag else "unknown"
            title = title_tag.get_text(strip=True) if title_tag else ""
            abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""
            year = year_tag.find("Year").get_text(strip=True) if year_tag and year_tag.find("Year") else "unknown"
            authors = ", ".join([a.get_text(strip=True) for a in author_tags[:3]])

            if abstract:  # skip articles with no abstract
                articles.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                    }
                )
        log.info(f"Fetched {len(articles)} abstracts from PubMed.")
        return articles
    except Exception as e:
        log.error(f"XML parsing of PubMed response failed: {e}")
        return []


def _fallback_rss_search(query: str, max_results: int = 10) -> list[dict]:
    """
    RSS-based fallback: query PubMed's RSS endpoint (no NCBI key required).
    Returns minimal article dicts.
    """
    log.info("Using PubMed RSS fallback...")
    rss_url = (
        f"https://pubmed.ncbi.nlm.nih.gov/rss/search/"
        f"?term={requests.utils.quote(query)}&count={max_results}"
    )
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:max_results]:
            articles.append(
                {
                    "pmid": entry.get("id", "rss_unknown"),
                    "title": entry.get("title", ""),
                    "abstract": entry.get("summary", ""),
                    "authors": "",
                    "year": entry.get("published", "")[:4],
                }
            )
        log.info(f"RSS fallback retrieved {len(articles)} articles.")
        return articles
    except Exception as e:
        log.error(f"RSS fallback failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Vector store integration
# ---------------------------------------------------------------------------

def _make_doc_id(pmid: str) -> str:
    return f"pubmed_{pmid}"


def index_articles(articles: list[dict]) -> None:
    """Embed and upsert articles into ChromaDB literature collection."""
    collection = get_or_create_collection(CHROMA_COLLECTION_LITERATURE)
    ids, texts, metas = [], [], []
    for art in articles:
        doc_id = _make_doc_id(art["pmid"])
        text = f"{art['title']}\n\n{art['abstract']}"
        ids.append(doc_id)
        texts.append(text)
        metas.append(
            {
                "pmid": art["pmid"],
                "title": art["title"][:200],
                "authors": art["authors"][:100],
                "year": art["year"],
            }
        )
    upsert_documents(collection, ids, texts, metas)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def search_literature(
    query: str,
    max_results: int = PUBMED_MAX_RESULTS,
    top_k: int = CHROMA_TOP_K,
    force_refresh: bool = False,
) -> list[dict]:
    """
    High-level literature search:
    1. Check ChromaDB for existing documents (skip fetching if populated).
    2. If empty or force_refresh, fetch from PubMed and index.
    3. Semantic search ChromaDB for query-relevant results.

    Returns a list of result dicts from ChromaDB (document, metadata, distance).
    """
    collection = get_or_create_collection(CHROMA_COLLECTION_LITERATURE)

    if collection.count() == 0 or force_refresh:
        log.info("Fetching literature from PubMed...")
        pmids = _entrez_search(query, max_results)
        if pmids:
            articles = _entrez_fetch_abstracts(pmids)
        else:
            articles = _fallback_rss_search(query, max_results)

        if not articles:
            log.warning("No literature retrieved from any source.")
            return []

        index_articles(articles)
        time.sleep(1)  # allow ChromaDB to settle

    results = query_collection(collection, query, top_k=top_k)
    log.info(f"Semantic search returned {len(results)} literature results.")
    return results


def extract_abstracts_text(results: list[dict]) -> list[str]:
    """Extract the raw document text from ChromaDB result dicts."""
    return [r["document"] for r in results if r.get("document")]