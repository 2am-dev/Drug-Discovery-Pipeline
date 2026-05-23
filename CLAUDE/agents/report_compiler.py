# drug_discovery_pipeline/agents/report_compiler.py
# =============================================================================
# FILE: agents/report_compiler.py
# ROLE: Report compilation agent.
#       Assembles all pipeline outputs into a structured Markdown report.
#       Uses the LLM to write an executive summary and polish each section.
#       Saves the final report to outputs/ with a timestamp.
# =============================================================================

import logging
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR
from utils.helpers import llm_call, save_text, timestamp, truncate
from utils.prompts import executive_summary_prompt, section_polish_prompt

log = logging.getLogger("drug_discovery.report_compiler")


class ReportCompilerAgent:
    """
    Compiles all agent outputs into a polished Markdown project proposal.

    Expects all populated state keys.
    Produces state keys: report, report_path
    """

    def run(self, state: dict) -> dict:
        log.info("ReportCompilerAgent running...")

        sections = []

        # ── 1. Executive Summary (LLM-generated) ───────────────────────
        exec_summary = self._write_executive_summary(state)
        sections.append(("Executive Summary", exec_summary))

        # ── 2. Research Plan ───────────────────────────────────────────
        plan_section = self._write_plan_section(state)
        sections.append(("Research Plan", plan_section))

        # ── 3. Literature & Patent Review ─────────────────────────────
        lit_section = self._write_literature_section(state)
        sections.append(("Literature & Patent Review", lit_section))

        # ── 4. Target Selection & Hypothesis ──────────────────────────
        hyp_section = self._write_hypothesis_section(state)
        sections.append(("Target Selection & Mechanistic Hypothesis", hyp_section))

        # ── 5. Molecule Design ─────────────────────────────────────────
        mol_section = self._write_molecule_section(state)
        sections.append(("De Novo Molecule Design", mol_section))

        # ── 6. Docking Evaluation ─────────────────────────────────────
        dock_section = self._write_docking_section(state)
        sections.append(("Molecular Docking Evaluation", dock_section))

        # ── 7. Synthetic Route ─────────────────────────────────────────
        synth_section = self._write_synthesis_section(state)
        sections.append(("Synthetic Feasibility & Proposed Routes", synth_section))

        # ── 8. Next Steps & Conclusion ─────────────────────────────────
        conclusion = self._write_conclusion(state)
        sections.append(("Conclusions & Next Steps", conclusion))

        # ── 9. Errors / Caveats ────────────────────────────────────────
        if state.get("errors"):
            errors_section = self._write_errors_section(state)
            sections.append(("Pipeline Notes & Caveats", errors_section))

        # ── Assemble markdown ─────────────────────────────────────────
        report = self._assemble_report(state, sections)
        state["report"] = report

        # ── Save to disk ──────────────────────────────────────────────
        fname = f"proposal_{state.get('run_id', timestamp())}.md"
        report_path = OUTPUT_DIR / fname
        save_text(report_path, report)
        state["report_path"] = str(report_path)

        log.info(f"Report saved: {report_path}")
        return state

    # ------------------------------------------------------------------
    # Section writers
    # ------------------------------------------------------------------

    def _write_executive_summary(self, state: dict) -> str:
        try:
            messages = executive_summary_prompt(state)
            return llm_call(messages)
        except Exception as e:
            log.error(f"Executive summary generation failed: {e}")
            return (
                f"This report presents AI-generated drug discovery findings for "
                f"**{state.get('input', 'the specified indication')}**. "
                f"(Executive summary generation encountered an error: {e})"
            )

    def _write_plan_section(self, state: dict) -> str:
        phases = state.get("plan", [])
        if not phases:
            return "_No plan data available._"
        lines = []
        for ph in phases:
            lines.append(
                f"**Phase {ph.get('phase_number', '?')}: {ph.get('phase_name', '')}**\n"
                f"{ph.get('description', '')}"
            )
            outputs = ph.get("expected_outputs", [])
            if outputs:
                lines.append("Expected outputs: " + "; ".join(outputs))
            lines.append("")
        return "\n".join(lines)

    def _write_literature_section(self, state: dict) -> str:
        lit = state.get("literature", {})
        patents = state.get("patents", {})
        synthesis = lit.get("synthesis", {})

        lines = ["### Literature Review\n"]
        lines.append(f"**Articles retrieved:** {lit.get('count', 0)}\n")

        if synthesis.get("summary"):
            lines.append(f"**State of the art:**\n{synthesis['summary']}\n")

        if synthesis.get("key_targets"):
            lines.append("**Key targets mentioned in literature:**")
            for t in synthesis["key_targets"][:8]:
                lines.append(f"  - {t}")
            lines.append("")

        if synthesis.get("known_mechanisms"):
            lines.append("**Mechanisms of action reported:**")
            for m in synthesis["known_mechanisms"][:6]:
                lines.append(f"  - {m}")
            lines.append("")

        if synthesis.get("research_gaps"):
            lines.append("**Identified research gaps:**")
            for g in synthesis["research_gaps"][:5]:
                lines.append(f"  - {g}")
            lines.append("")

        lines.append("### Patent Landscape\n")
        lines.append(f"**Patents indexed:** {patents.get('patent_count', 0)}\n")
        claims = patents.get("chemical_claims", [])
        if claims:
            lines.append("**Chemical entities extracted from patent claims:**")
            for c in claims[:5]:
                lines.append(f"  - `{c[:80]}`")

        polished = self._polish_section("Literature & Patent Review", "\n".join(lines))
        return polished

    def _write_hypothesis_section(self, state: dict) -> str:
        hyp = state.get("hypothesis", {})
        target = hyp.get("selected_target", {})

        lines = [
            f"### Selected Target\n",
            f"| Property | Value |",
            f"|---|---|",
            f"| Gene Name | **{target.get('gene_name', 'N/A')}** |",
            f"| UniProt ID | {target.get('uniprot_id', 'N/A')} |",
            f"| PDB ID | {target.get('pdb_id', 'N/A')} |",
            f"| Druggability Score | {hyp.get('druggability_score', 'N/A')} |",
            f"| Confidence | {hyp.get('confidence', 'N/A')} |",
            "",
            f"**Rationale:** {target.get('rationale', '')}",
            "",
            "### Mechanistic Hypothesis",
            "",
            f"> {hyp.get('hypothesis', 'Not available.')}",
            "",
            "### Justification",
            "",
            hyp.get("justification", ""),
        ]
        return "\n".join(lines)

    def _write_molecule_section(self, state: dict) -> str:
        candidates = state.get("candidates", {})
        stats = candidates.get("generation_stats", {})
        shortlist = candidates.get("shortlist", [])
        pharmacophore = state.get("pharmacophore", {})
        llm_commentary = candidates.get("llm_commentary", {})

        lines = [
            "### Generation Statistics\n",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total generated | {stats.get('total_generated', 'N/A')} |",
            f"| Valid molecules | {stats.get('valid_molecules', 'N/A')} |",
            f"| Drug-like (Lipinski+QED+SA) | {stats.get('drug_like', 'N/A')} |",
            f"| Shortlisted | {stats.get('shortlisted', 'N/A')} |",
            "",
            "### Pharmacophore Features\n",
        ]

        features = pharmacophore.get("pharmacophore_features", [])
        for f in features:
            lines.append(f"  - {f}")
        lines.append("")

        if pharmacophore.get("design_strategy"):
            lines.append(
                f"**Design strategy:** {pharmacophore['design_strategy']}\n"
            )

        lines.append("### Shortlisted Candidates\n")
        if shortlist:
            lines.append(
                "| # | SMILES | MW | LogP | QED | SA Score | TPSA |"
            )
            lines.append("|---|--------|-----|------|-----|----------|------|")
            for i, mol in enumerate(shortlist, 1):
                smi = mol.get("smiles", "")
                lines.append(
                    f"| {i} | `{smi[:50]}{'...' if len(smi)>50 else ''}` | "
                    f"{mol.get('MW', '')} | {mol.get('LogP', '')} | "
                    f"{mol.get('QED', '')} | {mol.get('SA_Score', '')} | "
                    f"{mol.get('TPSA', '')} |"
                )
            lines.append("")

        if llm_commentary and llm_commentary.get("design_notes"):
            lines.append(f"**Design notes:** {llm_commentary['design_notes']}")

        return "\n".join(lines)

    def _write_docking_section(self, state: dict) -> str:
        results = state.get("docking_results", [])
        interp = state.get("docking_interpretation", {})
        mode_note = ""

        lines = ["### Docking Results\n"]

        if not results:
            return "_No docking results available._"

        # Detect if mock was used
        mock_used = any(r.get("docking_mode", "").startswith("mock") for r in results)
        if mock_used:
            lines.append(
                "> ⚠️ **Note:** One or more docking scores are **mock/estimated** values "
                "because AutoDock Vina or the receptor PDBQT was unavailable. "
                "Replace with real Vina runs before making scientific decisions.\n"
            )

        lines.append(
            "| Rank | SMILES | Score (kcal/mol) | QED | MW | Mode |"
        )
        lines.append("|------|--------|------------------|-----|----|------|")
        for i, r in enumerate(results, 1):
            smi = r.get("smiles", "")
            lines.append(
                f"| {i} | `{smi[:45]}{'...' if len(smi)>45 else ''}` | "
                f"**{r.get('score', 'N/A')}** | "
                f"{r.get('QED', 'N/A')} | {r.get('MW', 'N/A')} | "
                f"{r.get('docking_mode', 'N/A')} |"
            )
        lines.append("")

        if interp.get("structural_insights"):
            lines.append(f"**Structural insights:** {interp['structural_insights']}\n")

        if interp.get("best_candidate"):
            lines.append(
                f"**Top candidate:** `{interp['best_candidate'][:80]}`"
            )

        return "\n".join(lines)

    def _write_synthesis_section(self, state: dict) -> str:
        synthesis = state.get("synthesis", [])
        if not synthesis:
            return "_No synthesis data available._"

        lines = []
        for i, res in enumerate(synthesis, 1):
            lines.append(f"### Candidate {i}: `{res.get('smiles', '')[:60]}`\n")
            lines.append(
                f"| Property | Value |\n|---|---|\n"
                f"| SA Score | {res.get('sa_score', 'N/A')} |\n"
                f"| SA Category | {res.get('sa_category', 'N/A')} |\n"
                f"| Estimated Steps | {res.get('estimated_steps', 'N/A')} |\n"
                f"| Feasibility | {res.get('feasibility', 'N/A')} |\n"
            )

            disc = res.get("rule_based_disconnections", [])
            if disc:
                lines.append("**Retrosynthetic disconnections:**")
                for d in disc:
                    lines.append(f"  {d}")
                lines.append("")

            llm_route = res.get("llm_route", {})
            if isinstance(llm_route, dict) and llm_route.get("forward_route"):
                lines.append("**Proposed forward synthetic route:**\n")
                for step in llm_route["forward_route"]:
                    lines.append(
                        f"  **Step {step.get('step', '?')}:** "
                        f"{step.get('reaction', '')} — "
                        f"Reagents: {step.get('reagents', '')} | "
                        f"Conditions: {step.get('conditions', '')} | "
                        f"Expected yield: ~{step.get('expected_yield_percent', '?')}%"
                    )
                lines.append("")

            if llm_route.get("key_challenges"):
                lines.append("**Key synthetic challenges:**")
                for ch in llm_route["key_challenges"]:
                    lines.append(f"  - {ch}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    def _write_conclusion(self, state: dict) -> str:
        hyp = state.get("hypothesis", {})
        target = hyp.get("selected_target", {})
        best_score = "N/A"
        best_smiles = "N/A"

        dr = state.get("docking_results", [])
        if dr:
            best_score = dr[0].get("score", "N/A")
            best_smiles = dr[0].get("smiles", "N/A")

        synth = state.get("synthesis", [])
        feasibility = synth[0].get("feasibility", "N/A") if synth else "N/A"

        text = f"""
### Summary of Key Findings

| Item | Detail |
|------|--------|
| Disease / Indication | {state.get('input', 'N/A')} |
| Selected Target | {target.get('gene_name', 'N/A')} ({target.get('uniprot_id', '')}) |
| PDB Structure | {target.get('pdb_id', 'N/A')} |
| Best Docking Score | {best_score} kcal/mol |
| Top Candidate SMILES | `{str(best_smiles)[:60]}` |
| Synthetic Feasibility | {feasibility} |

### Recommended Next Steps

1. **Experimental validation** – Test top candidates in biochemical assays
   (enzyme inhibition, binding assays, SPR).
2. **MD simulations** – Run 100 ns molecular dynamics on the top docking pose
   to assess binding stability.
3. **Analogue synthesis** – Synthesise 3-5 structurally related analogues to
   establish an initial SAR.
4. **ADMET profiling** – Assess absorption, distribution, metabolism, excretion
   and toxicity using cell-based assays and computational tools.
5. **Patent freedom-to-operate** – Conduct a detailed FTO analysis before
   entering synthesis.
6. **IND-enabling studies** – Plan in vivo efficacy studies in relevant disease
   models.

### Confidence Assessment

This proposal was generated using AI-driven methods. All computational
predictions (docking scores, SA scores, binding hypotheses) require
experimental validation. Docking scores marked as 'mock' are estimates
and should be replaced with real AutoDock Vina runs using a prepared
receptor PDBQT file.
"""
        return text.strip()

    def _write_errors_section(self, state: dict) -> str:
        errors = state.get("errors", [])
        if not errors:
            return ""
        lines = [
            "_The following non-fatal errors were recorded during the pipeline run:_\n"
        ]
        for e in errors:
            lines.append(f"- **{e.get('agent', 'Unknown')}** ({e.get('time', '')}): {e.get('error', '')}")
        return "\n".join(lines)

    def _polish_section(self, title: str, raw: str) -> str:
        """Ask the LLM to improve a section's language."""
        try:
            messages = section_polish_prompt(title, truncate(raw, 1500))
            return llm_call(messages)
        except Exception as e:
            log.warning(f"Section polishing failed for '{title}': {e}")
            return raw  # Return unpolished if LLM fails

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_report(
        self, state: dict, sections: list[tuple[str, str]]
    ) -> str:
        """Combine all sections into a full Markdown document."""
        run_id = state.get("run_id", "unknown")
        disease = state.get("input", "Unknown")
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        header = f"""# Drug Discovery Project Proposal
## {disease}

**Generated by:** End-to-End AI Drug Discovery Pipeline  
**Date:** {now}  
**Run ID:** `{run_id}`  
**LLM:** Ollama (local)  

---

"""
        toc_lines = ["## Table of Contents\n"]
        for i, (title, _) in enumerate(sections, 1):
            slug = title.lower().replace(" ", "-").replace("&", "").replace("/", "")
            toc_lines.append(f"{i}. [{title}](#{slug})")
        toc = "\n".join(toc_lines) + "\n\n---\n\n"

        body_parts = []
        for title, content in sections:
            body_parts.append(f"## {title}\n\n{content}\n\n---\n")

        footer = (
            "\n\n---\n*This report was automatically generated by the "
            "Drug Discovery AI Pipeline using open-source tools and a local "
            "Ollama LLM server. All findings are computational predictions "
            "and must be experimentally validated.*\n"
        )

        return header + toc + "\n".join(body_parts) + footer