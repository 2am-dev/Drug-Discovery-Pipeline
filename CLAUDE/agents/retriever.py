# drug_discovery_pipeline/agents/retriever.py
# =============================================================================
# FILE: agents/retriever.py
# ROLE: Literature and patent mining agent.
#       Orchestrates calls to literature_search, patent_search, and
#       target_lookup tools, then synthesises findings with the LLM.
# =============================================================================

import logging

from tools.literature_search import extract_abstracts_text, search_literature
from tools.patent_search import search_patents
from tools.target_lookup import lookup_target
from utils.helpers import llm_call, safe_extract_json, truncate
from utils.prompts import literature_synthesis_prompt

log = logging.getLogger("drug_discovery.retriever")


class RetrieverAgent:
    """
    Retrieves and synthesises scientific literature, patents, and
    protein target information relevant to the input query.

    Expects state keys: input
    Produces state keys: literature, patents, target_info
    """

    def run(self, state: dict) -> dict:
        query = state.get("input", "")
        log.info(f"RetrieverAgent running for query: '{query}'")

        # ── Literature search ─────────────────────────────────────────
        state["literature"] = self._retrieve_literature(query)

        # ── Patent search ─────────────────────────────────────────────
        state["patents"] = self._retrieve_patents(query)

        # ── Target lookup ─────────────────────────────────────────────
        state["target_info"] = self._retrieve_target(query)

        log.info("RetrieverAgent completed.")
        return state

    # ------------------------------------------------------------------

    def _retrieve_literature(self, query: str) -> dict:
        """Run PubMed search + LLM synthesis."""
        try:
            results = search_literature(query)
            abstracts = extract_abstracts_text(results)

            if not abstracts:
                log.warning("No literature abstracts retrieved.")
                return {"abstracts": [], "synthesis": {}, "raw_results": []}

            # LLM synthesis
            messages = literature_synthesis_prompt(query, abstracts)
            raw = llm_call(messages)
            synthesis = safe_extract_json(raw) or {"raw_text": raw}

            return {
                "abstracts": abstracts,
                "synthesis": synthesis,
                "raw_results": results,
                "count": len(abstracts),
            }
        except Exception as e:
            log.error(f"Literature retrieval failed: {e}")
            return {"abstracts": [], "synthesis": {}, "error": str(e)}

    def _retrieve_patents(self, query: str) -> dict:
        """Run patent search and return structured results."""
        try:
            result = search_patents(query)
            patent_texts = [r["document"] for r in result.get("results", [])]
            log.info(
                f"Retrieved {len(patent_texts)} relevant patents | "
                f"Chemical claims: {len(result.get('chemical_claims', []))}"
            )
            return result
        except Exception as e:
            log.error(f"Patent search failed: {e}")
            return {"results": [], "chemical_claims": [], "error": str(e)}

    def _retrieve_target(self, query: str) -> dict:
        """Look up protein targets on UniProt/PDB."""
        try:
            target_info = lookup_target(query)
            if target_info:
                log.info(
                    f"Target found: {target_info.get('gene_name')} | "
                    f"UniProt: {target_info.get('uniprot_id')} | "
                    f"PDB: {target_info.get('pdb_id')}"
                )
            else:
                log.warning("No target info retrieved from UniProt/PDB.")
            return target_info
        except Exception as e:
            log.error(f"Target lookup failed: {e}")
            return {"error": str(e)}