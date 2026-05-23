# drug_discovery_pipeline/utils/prompts.py
# =============================================================================
# FILE: utils/prompts.py
# ROLE: Centralised prompt-template library.
#       Unchanged in structure; json_repair_prompt added for the new
#       repair_json_with_llm() helper in helpers.py.
# =============================================================================

from typing import Any


def build_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user",   "content": user.strip()},
    ]


# ===========================================================================
# JSON repair  (routes to LOCAL deepseek-coder)
# ===========================================================================

JSON_REPAIR_SYSTEM = """
You are a JSON repair assistant. You receive malformed or truncated JSON text
and return ONLY the corrected, valid JSON — nothing else, no explanation.
"""

def json_repair_prompt(broken: str) -> list[dict]:
    return build_messages(
        JSON_REPAIR_SYSTEM,
        f"Fix this JSON and return only the corrected version:\n\n{broken[:3000]}"
    )


# ===========================================================================
# 1. PLANNER  →  REMOTE gemma4:31b
# ===========================================================================

PLANNER_SYSTEM = """
You are the chief scientific officer of an AI-driven drug discovery company.
You receive a disease indication or biological target as input and produce a
concise, structured research plan broken into numbered phases.

Respond ONLY with a valid JSON object (no markdown fences) with the key
"phases", whose value is a list of objects each containing:
  - "phase_number": int
  - "phase_name": str
  - "description": str
  - "expected_outputs": list[str]
"""

def planner_prompt(user_input: str) -> list[dict]:
    return build_messages(PLANNER_SYSTEM, f"""
Design a drug discovery research plan for:

INPUT: {user_input}

Cover: literature mining, patent analysis, target selection, hypothesis
generation, molecule design, docking evaluation, synthetic feasibility,
and report compilation. Return valid JSON only.
""")


# ===========================================================================
# 2. RETRIEVER / LITERATURE  →  LOCAL medgemma1.5
# ===========================================================================

RETRIEVER_SYSTEM = """
You are a biomedical literature analyst with deep expertise in pharmacology,
molecular biology, and medicinal chemistry.  You receive PubMed abstracts
and patent summaries and extract the most relevant scientific intelligence.
"""

def literature_synthesis_prompt(query: str, abstracts: list[str]) -> list[dict]:
    combined = "\n\n---\n\n".join(
        [f"[{i+1}] {a}" for i, a in enumerate(abstracts[:15])]
    )
    return build_messages(RETRIEVER_SYSTEM, f"""
RESEARCH QUERY: {query}

Analyse the abstracts below. Return a JSON object (no fences) with:
  - "key_targets": list of mentioned biological targets/proteins
  - "known_mechanisms": list of described mechanisms of action
  - "known_compounds": list of drug names or SMILES
  - "research_gaps": list of open questions
  - "summary": 3-5 sentence synthesis of the state-of-the-art

ABSTRACTS:
{combined}
""")


# ===========================================================================
# 3. HYPOTHESIS  →  REMOTE gemma4:31b
# ===========================================================================

HYPOTHESIS_SYSTEM = """
You are a senior medicinal chemist and molecular biologist. You evaluate
evidence from literature and patents to select the most druggable target
and articulate a clear, testable mechanistic hypothesis for a new therapeutic.
"""

def hypothesis_prompt(
    disease: str,
    lit_findings: dict,
    patent_findings: dict,
    target_candidates: list[dict],
) -> list[dict]:
    return build_messages(HYPOTHESIS_SYSTEM, f"""
DISEASE / INDICATION: {disease}

LITERATURE FINDINGS:
{lit_findings}

PATENT LANDSCAPE:
{patent_findings}

TARGET CANDIDATES (UniProt/PDB):
{target_candidates}

Return a JSON object (no fences) with:
  - "selected_target": {{
        "gene_name": str, "uniprot_id": str, "pdb_id": str,
        "rationale": str
    }}
  - "hypothesis": str  (≤100 words, mechanistic)
  - "justification": str  (3-5 sentences)
  - "druggability_score": float 0-1
  - "confidence": "high"|"medium"|"low"
""")


# ===========================================================================
# 4. MOLECULE DESIGNER  →  REMOTE gemma4:31b
# ===========================================================================

MOLECULE_DESIGNER_SYSTEM = """
You are a computational medicinal chemist specialising in structure-activity
relationships and de novo drug design.
"""

def pharmacophore_prompt(
    target: dict,
    known_binders: list[str],
    pocket_description: str,
) -> list[dict]:
    binders_str = "\n".join(known_binders[:10]) if known_binders else "None available"
    return build_messages(MOLECULE_DESIGNER_SYSTEM, f"""
TARGET: {target.get('gene_name','Unknown')} ({target.get('uniprot_id','')})
BINDING POCKET: {pocket_description}

KNOWN BINDERS:
{binders_str}

Extract pharmacophoric features. Return JSON (no fences) with:
  - "pharmacophore_features": list[str]
  - "core_scaffolds": list[str] (SMILES)
  - "forbidden_groups": list[str]
  - "design_strategy": str
""")


def molecule_refinement_prompt(
    smiles_list: list[str],
    target: dict,
    pharmacophore: dict,
) -> list[dict]:
    return build_messages(MOLECULE_DESIGNER_SYSTEM, f"""
TARGET: {target.get('gene_name','Unknown')}
PHARMACOPHORE: {pharmacophore}

CANDIDATES:
{chr(10).join(smiles_list)}

Return JSON (no fences) with:
  - "top_candidates": list of {{
        "smiles": str,
        "rationale": str,
        "predicted_activity": "high"|"medium"|"low"
    }}
  - "design_notes": str
""")


# ===========================================================================
# 5. DOCKING EVALUATOR  →  REMOTE gemma4:31b
# ===========================================================================

DOCKING_SYSTEM = """
You are a structural bioinformatician who interprets molecular docking results
and relates binding energies to likely biological activity.
"""

def docking_interpretation_prompt(
    target: dict,
    docking_results: list[dict],
) -> list[dict]:
    results_str = "\n".join(
        [f"  {r['smiles'][:50]}… → {r['score']} kcal/mol" for r in docking_results]
    )
    return build_messages(DOCKING_SYSTEM, f"""
TARGET: {target.get('gene_name','Unknown')} (PDB: {target.get('pdb_id','N/A')})

DOCKING RESULTS:
{results_str}

Return JSON (no fences) with:
  - "ranked_compounds": list of {{
        "smiles": str, "score_kcal_mol": float,
        "interpretation": str, "predicted_ic50_nm": str
    }}
  - "best_candidate": str (SMILES)
  - "structural_insights": str (2-3 sentences)
""")


# ===========================================================================
# 6. SYNTHESIS EVALUATOR  →  REMOTE gemma4:31b
# ===========================================================================

SYNTHESIS_SYSTEM = """
You are an expert synthetic organic chemist with experience in medicinal
chemistry. You propose practical retrosynthetic routes considering commercial
availability, reagent costs, and reaction robustness.
"""

def retrosynthesis_prompt(smiles: str, sa_score: float, target: dict) -> list[dict]:
    return build_messages(SYNTHESIS_SYSTEM, f"""
DRUG CANDIDATE SMILES: {smiles}
SA SCORE: {sa_score:.2f}  (1=easy, 10=hard)
TARGET: {target.get('gene_name','Unknown')}

Return JSON (no fences) with:
  - "retrosynthetic_steps": list[str]
  - "forward_route": list of {{
        "step": int, "reaction": str, "reagents": str,
        "conditions": str, "expected_yield_percent": int
    }}
  - "starting_materials": list[str]
  - "estimated_steps": int
  - "feasibility": "straightforward"|"moderate"|"challenging"
  - "key_challenges": list[str]
""")


# ===========================================================================
# 7. REPORT COMPILER
#    executive_summary / conclusion  →  REMOTE gemma4:31b
#    section_polish                  →  LOCAL  mistral:7b
# ===========================================================================

REPORT_SYSTEM = """
You are a scientific writer and drug discovery project lead. You compile
technical findings into a polished, publication-quality Project Proposal.
"""

def executive_summary_prompt(state: dict) -> list[dict]:
    dr = state.get("docking_results", [])
    best_score = dr[0].get("score", "N/A") if dr else "N/A"
    best_smi   = dr[0].get("smiles", "N/A") if dr else "N/A"
    return build_messages(REPORT_SYSTEM, f"""
Write a 300-400 word executive summary for a drug discovery project proposal.

INPUT: {state.get('input','')}
SELECTED TARGET: {state.get('hypothesis',{}).get('selected_target',{})}
HYPOTHESIS: {state.get('hypothesis',{}).get('hypothesis','')}
BEST CANDIDATE SMILES: {best_smi}
TOP DOCKING SCORE: {best_score} kcal/mol

Include: scientific rationale, target justification, key in silico findings,
and next steps. Write for a pharmaceutical R&D audience.
""")


def section_polish_prompt(section_title: str, raw_content: str) -> list[dict]:
    return build_messages(REPORT_SYSTEM, f"""
Polish this draft section. Improve clarity and flow. Keep facts unchanged.
Max 500 words. Return only the polished text — no JSON, no code fences.

SECTION: {section_title}

DRAFT:
{raw_content}
""")