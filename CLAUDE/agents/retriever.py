# drug_discovery_pipeline/agents/retriever.py
# =============================================================================
# FILE: agents/retriever.py
# ROLE: Literature and patent mining agent.
#       Literature LLM synthesis → LOCAL medgemma1.5  (biomedical specialist)
#       All other logic unchanged.
# =============================================================================

import logging

from tools.literature_search import extract_abstracts_text, search_literature
from tools.patent_search import search_patents
from tools.target_lookup import lookup_target
from utils.helpers import routed_llm_call, safe_extract_json
from utils.prompts import literature_synthesis_prompt

log = logging.getLogger("drug_discovery.retriever")


class RetrieverAgent:
    """
    Literature synthesis → LOCAL medgemma1.5  (task="literature_synthesis")
    Patent + target lookup → pure API calls (no LLM needed)
    """

    def run(self, state: dict) -> dict:
        query = state.get("input", "")
        log.info(f"RetrieverAgent: '{query}'")
        state["literature"] = self._retrieve_literature(query)
        state["patents"]     = self._retrieve_patents(query)
        state["target_info"] = self._retrieve_target(query)
        return state

    def _retrieve_literature(self, query: str) -> dict:
        try:
            results   = search_literature(query)
            abstracts = extract_abstracts_text(results)
            if not abstracts:
                log.warning("No abstracts retrieved.")
                return {"abstracts": [], "synthesis": {}, "raw_results": []}

            messages  = literature_synthesis_prompt(query, abstracts)
            # ← medgemma1.5 on LOCAL server
            raw       = routed_llm_call("literature_synthesis", messages)
            synthesis = safe_extract_json(raw) or {"raw_text": raw}
            return {
                "abstracts": abstracts, "synthesis": synthesis,
                "raw_results": results, "count": len(abstracts),
            }
        except Exception as e:
            log.error(f"Literature retrieval failed: {e}")
            return {"abstracts": [], "synthesis": {}, "error": str(e)}

    def _retrieve_patents(self, query: str) -> dict:
        try:
            result = search_patents(query)
            log.info(
                f"{len(result.get('results',[]))} patents | "
                f"{len(result.get('chemical_claims',[]))} claims"
            )
            return result
        except Exception as e:
            log.error(f"Patent search failed: {e}")
            return {"results": [], "chemical_claims": [], "error": str(e)}

    def _retrieve_target(self, query: str) -> dict:
        try:
            info = lookup_target(query)
            if info:
                log.info(
                    f"Target: {info.get('gene_name')} | "
                    f"{info.get('uniprot_id')} | PDB {info.get('pdb_id')}"
                )
            return info or {}
        except Exception as e:
            log.error(f"Target lookup failed: {e}")
            return {"error": str(e)}