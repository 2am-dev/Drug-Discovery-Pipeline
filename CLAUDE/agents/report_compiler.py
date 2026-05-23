# drug_discovery_pipeline/agents/report_compiler.py
# =============================================================================
# FILE: agents/report_compiler.py
# ROLE: Report compilation agent.
#       Executive summary / conclusion → REMOTE gemma4:31b
#       Section polishing              → LOCAL  mistral:7b  (fast, good enough)
#       Assembly                       → pure Python string ops
# =============================================================================

import logging
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR
from utils.helpers import routed_llm_call, save_text, timestamp, truncate
from utils.prompts import executive_summary_prompt, section_polish_prompt

log = logging.getLogger("drug_discovery.report_compiler")


class ReportCompilerAgent:
    """
    Heavy LLM tasks  → REMOTE gemma4:31b   (executive_summary, conclusion)
    Section polishing → LOCAL mistral:7b   (section_polish)
    """

    def run(self, state: dict) -> dict:
        log.info("ReportCompilerAgent running...")
        sections = []

        # ① Executive summary — REMOTE
        sections.append(("Executive Summary",            self._exec_summary(state)))
        # ② Plan
        sections.append(("Research Plan",                self._plan(state)))
        # ③ Literature — POLISHED locally
        sections.append(("Literature & Patent Review",   self._literature(state)))
        # ④ Hypothesis
        sections.append(("Target Selection & Hypothesis", self._hypothesis(state)))
        # ⑤ Molecules
        sections.append(("De Novo Molecule Design",      self._molecules(state)))
        # ⑥ Docking
        sections.append(("Molecular Docking Evaluation", self._docking(state)))
        # ⑦ Synthesis
        sections.append(("Synthetic Feasibility",        self._synthesis(state)))
        # ⑧ Conclusion — REMOTE
        sections.append(("Conclusions & Next Steps",     self._conclusion(state)))
        # ⑨ Errors
        if state.get("errors"):
            sections.append(("Pipeline Notes & Caveats", self._errors(state)))

        report = self._assemble(state, sections)
        state["report"] = report

        fname = f"proposal_{state.get('run_id', timestamp())}.md"
        path  = OUTPUT_DIR / fname
        save_text(path, report)
        state["report_path"] = str(path)
        log.info(f"Report saved: {path}")
        return state

    # ------------------------------------------------------------------
    # LLM-heavy sections
    # ------------------------------------------------------------------

    def _exec_summary(self, state: dict) -> str:
        try:
            msgs = executive_summary_prompt(state)
            return routed_llm_call("executive_summary", msgs)   # ← REMOTE
        except Exception as e:
            log.error(f"Executive summary failed: {e}")
            return f"Executive summary unavailable (error: {e})"

    def _conclusion(self, state: dict) -> str:
        hyp    = state.get("hypothesis", {})
        target = hyp.get("selected_target", {})
        dr     = state.get("docking_results", [])
        synth  = state.get("synthesis", [])

        best_score = dr[0].get("score","N/A") if dr else "N/A"
        best_smi   = dr[0].get("smiles","N/A") if dr else "N/A"
        feasibility = synth[0].get("feasibility","N/A") if synth else "N/A"

        raw = f"""
| Item | Detail |
|---|---|
| Disease | {state.get('input','N/A')} |
| Target | {target.get('gene_name','N/A')} ({target.get('uniprot_id','')}) |
| PDB | {target.get('pdb_id','N/A')} |
| Best Docking Score | {best_score} kcal/mol |
| Top Candidate | `{str(best_smi)[:60]}` |
| Synthetic Feasibility | {feasibility} |

**Recommended next steps:**
1. Biochemical assay validation (enzyme inhibition, SPR binding).
2. 100 ns MD simulation on the best docking pose.
3. Synthesis of 3-5 analogues to establish initial SAR.
4. ADMET profiling (computational + cell-based).
5. Freedom-to-operate patent analysis.
6. IND-enabling in vivo efficacy studies.

All computational predictions require experimental validation.
Docking scores marked 'mock' must be replaced with real Vina runs.
"""
        try:
            from utils.prompts import build_messages, REPORT_SYSTEM
            msgs = build_messages(
                REPORT_SYSTEM,
                f"Polish and expand this conclusions section into prose "
                f"(keep the table):\n\n{raw}"
            )
            return routed_llm_call("conclusion", msgs)   # ← REMOTE
        except Exception as e:
            log.warning(f"Conclusion polish failed: {e}")
            return raw.strip()

    # ------------------------------------------------------------------
    # Locally-polished sections (mistral:7b)
    # ------------------------------------------------------------------

    def _polish(self, title: str, raw: str) -> str:
        try:
            msgs = section_polish_prompt(title, truncate(raw, 1500))
            return routed_llm_call("section_polish", msgs)   # ← LOCAL mistral:7b
        except Exception as e:
            log.warning(f"Polish failed for '{title}': {e}")
            return raw

    # ------------------------------------------------------------------
    # Pure-Python section builders (no LLM needed)
    # ------------------------------------------------------------------

    def _plan(self, state: dict) -> str:
        phases = state.get("plan", [])
        if not phases:
            return "_No plan data._"
        lines = []
        for ph in phases:
            lines.append(
                f"**Phase {ph.get('phase_number','?')}: {ph.get('phase_name','')}**\n"
                f"{ph.get('description','')}"
            )
            for o in ph.get("expected_outputs", []):
                lines.append(f"  - {o}")
            lines.append("")
        return "\n".join(lines)

    def _literature(self, state: dict) -> str:
        lit  = state.get("literature", {})
        pat  = state.get("patents", {})
        syn  = lit.get("synthesis", {})
        lines = [
            f"**Articles retrieved:** {lit.get('count',0)}\n",
            f"**Summary:** {syn.get('summary','')}\n",
        ]
        if syn.get("key_targets"):
            lines.append("**Key targets:**")
            for t in syn["key_targets"][:6]: lines.append(f"  - {t}")
        if syn.get("research_gaps"):
            lines.append("\n**Research gaps:**")
            for g in syn["research_gaps"][:4]: lines.append(f"  - {g}")
        lines += [
            f"\n**Patents indexed:** {pat.get('patent_count',0)}",
        ]
        claims = pat.get("chemical_claims", [])
        if claims:
            lines.append("**Chemical claims extracted:**")
            for c in claims[:5]: lines.append(f"  - `{c[:80]}`")
        raw = "\n".join(lines)
        return self._polish("Literature & Patent Review", raw)

    def _hypothesis(self, state: dict) -> str:
        hyp = state.get("hypothesis", {})
        tgt = hyp.get("selected_target", {})
        return "\n".join([
            "### Selected Target\n",
            "| Property | Value |", "|---|---|",
            f"| Gene | **{tgt.get('gene_name','N/A')}** |",
            f"| UniProt | {tgt.get('uniprot_id','N/A')} |",
            f"| PDB | {tgt.get('pdb_id','N/A')} |",
            f"| Druggability | {hyp.get('druggability_score','N/A')} |",
            f"| Confidence | {hyp.get('confidence','N/A')} |",
            f"\n**Rationale:** {tgt.get('rationale','')}",
            "\n### Mechanistic Hypothesis\n",
            f"> {hyp.get('hypothesis','Not available.')}",
            f"\n### Justification\n{hyp.get('justification','')}",
        ])

    def _molecules(self, state: dict) -> str:
        cands  = state.get("candidates", {})
        stats  = cands.get("generation_stats", {})
        short  = cands.get("shortlist", [])
        pharma = state.get("pharmacophore", {})
        lines  = [
            "### Generation Stats\n",
            "| Metric | Value |", "|---|---|",
            f"| Generated | {stats.get('total_generated','N/A')} |",
            f"| Valid     | {stats.get('valid_molecules','N/A')} |",
            f"| Drug-like | {stats.get('drug_like','N/A')} |",
            f"| Shortlist | {stats.get('shortlisted','N/A')} |",
            "\n### Pharmacophore\n",
        ]
        for f in pharma.get("pharmacophore_features", []): lines.append(f"  - {f}")
        if pharma.get("design_strategy"):
            lines.append(f"\n**Strategy:** {pharma['design_strategy']}\n")
        lines.append("\n### Shortlisted Candidates\n")
        if short:
            lines.append("| # | SMILES | MW | LogP | QED | SA |")
            lines.append("|---|--------|-----|------|-----|----|")
            for i, m in enumerate(short, 1):
                s = m.get("smiles","")
                lines.append(
                    f"| {i} | `{s[:45]}{'...' if len(s)>45 else ''}` | "
                    f"{m.get('MW','')} | {m.get('LogP','')} | "
                    f"{m.get('QED','')} | {m.get('SA_Score','')} |"
                )
        return "\n".join(lines)

    def _docking(self, state: dict) -> str:
        results = state.get("docking_results", [])
        interp  = state.get("docking_interpretation", {})
        if not results:
            return "_No docking results._"
        lines = []
        mock = any(r.get("docking_mode","").startswith("mock") for r in results)
        if mock:
            lines.append(
                "> ⚠️ **Mock scores** — Vina/PDBQT unavailable. "
                "Replace before scientific use.\n"
            )
        lines += ["| Rank | SMILES | Score (kcal/mol) | QED | Mode |",
                  "|------|--------|------------------|-----|------|"]
        for i, r in enumerate(results, 1):
            s = r.get("smiles","")
            lines.append(
                f"| {i} | `{s[:40]}{'...' if len(s)>40 else ''}` | "
                f"**{r.get('score','N/A')}** | {r.get('QED','N/A')} | "
                f"{r.get('docking_mode','N/A')} |"
            )
        if interp.get("structural_insights"):
            lines.append(f"\n**Structural insights:** {interp['structural_insights']}")
        if interp.get("best_candidate"):
            lines.append(f"\n**Top candidate:** `{interp['best_candidate'][:80]}`")
        return "\n".join(lines)

    def _synthesis(self, state: dict) -> str:
        synth = state.get("synthesis", [])
        if not synth:
            return "_No synthesis data._"
        lines = []
        for i, res in enumerate(synth, 1):
            lines += [
                f"### Candidate {i}: `{res.get('smiles','')[:60]}`\n",
                "| Property | Value |", "|---|---|",
                f"| SA Score | {res.get('sa_score','N/A')} |",
                f"| Category | {res.get('sa_category','N/A')} |",
                f"| Est. Steps | {res.get('estimated_steps','N/A')} |",
                f"| Feasibility | {res.get('feasibility','N/A')} |",
                "",
            ]
            for d in res.get("rule_based_disconnections", []):
                lines.append(f"  {d}")
            route = res.get("llm_route", {})
            if isinstance(route, dict):
                for step in route.get("forward_route", []):
                    lines.append(
                        f"  **Step {step.get('step','?')}:** "
                        f"{step.get('reaction','')} | "
                        f"{step.get('reagents','')} | "
                        f"~{step.get('expected_yield_percent','?')}%"
                    )
            lines.append("---\n")
        return "\n".join(lines)

    def _errors(self, state: dict) -> str:
        errs = state.get("errors", [])
        lines = ["_Non-fatal pipeline errors:_\n"]
        for e in errs:
            lines.append(f"- **{e.get('agent','')}**: {e.get('error','')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, state: dict, sections: list[tuple[str,str]]) -> str:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        hdr = (
            f"# Drug Discovery Project Proposal\n"
            f"## {state.get('input','Unknown')}\n\n"
            f"**Date:** {now}  \n"
            f"**Run ID:** `{state.get('run_id','?')}`  \n"
            f"**Primary LLM:** gemma4:31b-it-q8_0 (REMOTE A6000)  \n"
            f"**Bio LLM:** medgemma1.5 (LOCAL 5070 Ti)  \n"
            f"**Code LLM:** deepseek-coder:6.7b (LOCAL 5070 Ti)  \n\n---\n\n"
        )
        toc = "## Table of Contents\n\n"
        for i, (title, _) in enumerate(sections, 1):
            slug = title.lower().replace(" ","‑").replace("&","").replace("/","")
            toc += f"{i}. [{title}](#{slug})\n"
        toc += "\n---\n\n"

        body = "\n".join(f"## {t}\n\n{c}\n\n---\n" for t, c in sections)
        footer = (
            "\n\n---\n*Generated by AI Drug Discovery Pipeline. "
            "Computational predictions only — experimental validation required.*\n"
        )
        return hdr + toc + body + footer