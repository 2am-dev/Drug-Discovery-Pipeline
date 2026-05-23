# drug_discovery_pipeline/agents/hypothesis.py
"""
Hypothesis Agent – Target Selection & Mechanistic Hypothesis.

1. Looks up candidate targets from UniProt.
2. Uses the LLM to rank them and formulate a hypothesis.
3. Resolves the best available PDB structure.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

from utils.helpers import call_llm
from utils.prompts import HYPOTHESIS_SYSTEM, HYPOTHESIS_PROMPT
from tools.target_lookup import lookup_target, get_pdb_structure

logger = logging.getLogger(__name__)


class HypothesisAgent:
    """Selects a druggable target and formulates a mechanistic hypothesis."""

    def run(self, state: Dict) -> Dict:
        query = state.get("input", "")
        logger.info("HypothesisAgent starting for: '%s'", query)

        # ── Look up targets ──────────────────────────────────────────────
        targets = lookup_target(query, limit=5)
        if not targets:
            # Try broader search
            targets = lookup_target(query.split()[0], limit=5)

        # Also search for each literature-mentioned gene if available
        target_text = ""
        if targets:
            for t in targets:
                pdb_str = ", ".join(t["pdb_ids"][:3]) if t["pdb_ids"] else "N/A"
                target_text += (
                    f"- {t['gene']} ({t['accession']}) – {t['protein_name']} "
                    f"[{t['organism']}] PDB: {pdb_str}\n"
                )
        else:
            target_text = "No targets found via UniProt – please specify a target gene."

        # ── Format literature & patents for prompt ───────────────────────
        lit_text = ""
        for art in state.get("literature_results", [])[:8]:
            lit_text += f"[PMID: {art.get('pmid', '?')}] {art.get('title', '')}\n{art.get('text', '')[:300]}\n\n"

        pat_text = ""
        for p in state.get("patent_results", [])[:5]:
            pat_text += f"[Patent: {p.get('patent_number', '?')}] {p.get('title', '')} ({p.get('date', '')})\n{(p.get('abstract', '') or '')[:300]}\n\n"

        # ── LLM hypothesis generation ────────────────────────────────────
        prompt = HYPOTHESIS_PROMPT.format(
            input=query,
            literature=lit_text or "No literature available.",
            patents=pat_text or "No patent data available.",
            targets=target_text,
        )
        response = call_llm(prompt, system=HYPOTHESIS_SYSTEM, temperature=0.4, max_tokens=1024)

        # ── Parse response ───────────────────────────────────────────────
        parsed = self._parse_response(response)
        parsed["raw_response"] = response

        # Resolve PDB ID if missing
        if parsed.get("pdb_id", "N/A") == "N/A" and parsed.get("uniprot_id"):
            pdb = get_pdb_structure(parsed["uniprot_id"])
            if pdb:
                parsed["pdb_id"] = pdb

        state["target_info"] = parsed
        state["hypothesis"] = {
            "target": parsed.get("selected_target", ""),
            "uniprot_id": parsed.get("uniprot_id", ""),
            "pdb_id": parsed.get("pdb_id", ""),
            "hypothesis": parsed.get("hypothesis", ""),
            "justification": parsed.get("justification", ""),
        }

        logger.info(
            "HypothesisAgent done – target: %s, PDB: %s",
            parsed.get("selected_target", "?"),
            parsed.get("pdb_id", "N/A"),
        )
        return state

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Extract structured fields from the LLM's YAML-like response."""
        result: dict = {
            "selected_target": "",
            "uniprot_id": "",
            "pdb_id": "N/A",
            "hypothesis": "",
            "justification": "",
        }
        # Simple key: value parsing
        patterns = {
            "selected_target": r"selected_target:\s*(.+)",
            "uniprot_id": r"uniprot_id:\s*(.+)",
            "pdb_id": r"pdb_id:\s*(.+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                result[key] = m.group(1).strip()

        # Multi-line fields (hypothesis, justification)
        for field in ("hypothesis", "justification"):
            m = re.search(rf"{field}:\s*\|?\s*\n((?:  .+\n?)+)", text)
            if m:
                result[field] = " ".join(line.strip() for line in m.group(1).strip().splitlines())

        return result