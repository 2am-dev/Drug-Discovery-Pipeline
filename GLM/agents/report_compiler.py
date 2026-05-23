# drug_discovery_pipeline/agents/report_compiler.py
"""
Report Compiler Agent – assembles all sections into a polished Markdown
report and saves it to the outputs directory.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Dict

from utils.helpers import call_llm, sanitize_filename
from utils.prompts import REPORT_SYSTEM, REPORT_PROMPT
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class ReportCompilerAgent:
    """Compiles the final Project Proposal Report in Markdown."""

    def run(self, state: Dict) -> Dict:
        logger.info("ReportCompilerAgent starting…")

        # ── Prepare section texts ────────────────────────────────────────
        input_text = state.get("input", "")

        lit_results = state.get("literature_results", [])
        literature_text = self._format_literature(lit_results)

        pat_results = state.get("patent_results", [])
        patents_text = self._format_patents(pat_results)

        hypothesis = state.get("hypothesis", {})
        target_text = (
            f"**Target:** {hypothesis.get('target', 'N/A')}  \n"
            f"**UniProt:** {hypothesis.get('uniprot_id', 'N/A')}  \n"
            f"**PDB:** {hypothesis.get('pdb_id', 'N/A')}  \n"
        )
        hypothesis_text = (
            f"**Hypothesis:** {hypothesis.get('hypothesis', 'N/A')}  \n\n"
            f"**Justification:** {hypothesis.get('justification', 'N/A')}"
        )

        pharmacophore = state.get("pharmacophore", "N/A")

        molecules_text = self._format_molecules(state)

        docking_text = self._format_docking(state.get("docking_results", []))

        synthesis_text = self._format_synthesis(state.get("synthesis_results", []))

        # ── Call LLM to compile report ───────────────────────────────────
        prompt = REPORT_PROMPT.format(
            input=input_text,
            literature=literature_text,
            patents=patents_text,
            target=target_text,
            hypothesis=hypothesis_text,
            pharmacophore=pharmacophore,
            molecules=molecules_text,
            docking=docking_text,
            synthesis=synthesis_text,
        )
        report_md = call_llm(prompt, system=REPORT_SYSTEM, temperature=0.4, max_tokens=6000)

        # ── Save to file ─────────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_filename(input_text)[:40]
        filename = f"proposal_{safe_name}_{ts}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_md)

        state["report_md"] = report_md
        state["report_path"] = filepath

        logger.info("Report saved to: %s", filepath)
        return state

    # ── Formatting helpers ────────────────────────────────────────────────

    @staticmethod
    def _format_literature(articles: list) -> str:
        if not articles:
            return "No literature results available."
        lines: list[str] = []
        for a in articles[:10]:
            lines.append(f"- [PMID: {a.get('pmid', '?')}] {a.get('title', '')}")
        return "\n".join(lines)

    @staticmethod
    def _format_patents(patents: list) -> str:
        if not patents:
            return "No patent results available."
        lines: list[str] = []
        for p in patents[:8]:
            lines.append(f"- [Patent: {p.get('patent_number', '?')}] {p.get('title', '')} ({p.get('date', '')})")
        return "\n".join(lines)

    @staticmethod
    def _format_molecules(state: dict) -> str:
        lines: list[str] = []
        generated = state.get("generated_molecules", [])
        lines.append(f"Total generated: {len(generated)}")
        shortlisted = state.get("shortlisted_molecules", [])
        lines.append(f"Shortlisted: {len(shortlisted)}\n")
        for i, m in enumerate(shortlisted, 1):
            lines.append(
                f"{i}. `{m['smiles']}`  – QED: {m.get('qed', 0):.3f}, "
                f"SA: {m.get('sa_score', 0):.2f}, "
                f"Composite: {m.get('composite_score', 0):.3f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_docking(results: list) -> str:
        if not results:
            return "No docking results available."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. `{r['smiles']}` – Affinity: {r.get('affinity_kcal_mol', 0):.2f} kcal/mol "
                f"(method: {r.get('method', '?')})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_synthesis(results: list) -> str:
        if not results:
            return "No synthesis results available."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"### Molecule {i}: `{r['smiles']}`\n"
                f"- SA Score: {r.get('sa_score', 0):.2f} ({r.get('feasibility', '?')})\n"
                f"- Docking: {r.get('affinity_kcal_mol', 0):.2f} kcal/mol\n\n"
                f"**Retrosynthetic Route:**\n{r.get('retrosynthetic_route', 'N/A')}\n"
            )
        return "\n".join(lines)