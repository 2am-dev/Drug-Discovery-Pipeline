# drug_discovery_pipeline/agents/retriever.py
"""
Retriever Agent – Literature & Patent Mining.

Uses the LLM to expand the user query into focused search terms, then
delegates to the literature and patent search tools.
"""

from __future__ import annotations

import logging
from typing import Dict

from utils.helpers import call_llm
from utils.prompts import SEARCH_QUERY_PROMPT, PATENT_QUERY_PROMPT
from tools.literature_search import search_literature
from tools.patent_search import search_patents

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Mines scientific literature and patents for the given topic."""

    def run(self, state: Dict) -> Dict:
        query = state.get("input", "")
        logger.info("RetrieverAgent starting for: '%s'", query)

        # ── Generate search queries via LLM ──────────────────────────────
        lit_queries_text = call_llm(
            SEARCH_QUERY_PROMPT.format(input=query),
            temperature=0.3,
            max_tokens=256,
        )
        lit_queries = [q.strip() for q in lit_queries_text.strip().splitlines() if q.strip()]
        if not lit_queries:
            lit_queries = [query]

        pat_queries_text = call_llm(
            PATENT_QUERY_PROMPT.format(input=query),
            temperature=0.3,
            max_tokens=128,
        )
        pat_queries = [q.strip() for q in pat_queries_text.strip().splitlines() if q.strip()]
        if not pat_queries:
            pat_queries = [query]

        # ── Literature search ────────────────────────────────────────────
        all_lit: list[dict] = []
        for q in lit_queries:
            results = search_literature(q, top_k=8)
            all_lit.extend(results)
        # Deduplicate by pmid
        seen_pmids: set[str] = set()
        unique_lit: list[dict] = []
        for art in all_lit:
            pmid = art.get("pmid", "")
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_lit.append(art)

        # ── Patent search ────────────────────────────────────────────────
        all_pat: list[dict] = []
        for q in pat_queries:
            results = search_patents(q, top_k=8)
            all_pat.extend(results)
        # Deduplicate by patent_number
        seen_pn: set[str] = set()
        unique_pat: list[dict] = []
        for p in all_pat:
            pn = p.get("patent_number", "")
            if pn not in seen_pn:
                seen_pn.add(pn)
                unique_pat.append(p)

        state["literature_results"] = unique_lit
        state["patent_results"] = unique_pat

        logger.info(
            "RetrieverAgent done – %d literature, %d patents.",
            len(unique_lit), len(unique_pat),
        )
        return state