# drug_discovery_pipeline/utils/prompts.py
# =============================================================================
# FILE: utils/prompts.py
# ROLE: Centralised prompt-template library.
#       Every LLM call in the project fetches its system + user prompt from
#       here.  Keeping prompts in one place makes iteration fast and keeps
#       agent code clean.
# =============================================================================

from typing import Optional


# ---------------------------------------------------------------------------
# Helper – build a standard chat message list
# ---------------------------------------------------------------------------

def build_messages(system: str, user: str) -> list[dict]:
    """Return an OpenAI-compatible message list."""
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user",   "content": user.strip()},
    ]


# ===========================================================================
# 1. PLANNER AGENT
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
    user = f"""
Design a drug discovery research plan for the following input:

INPUT: {user_input}

Return a JSON plan covering: literature mining, patent analysis, target
selection, hypothesis generation, molecule design, docking evaluation,
synthetic feasibility, and report compilation.
"""
    return build_messages(PLANNER_SYSTEM, user)


# ===========================================================================
# 2. RETRIEVER / LITERATURE AGENT
# ===========================================================================

RETRIEVER_SYSTEM = """
You are a biomedical literature analyst with deep expertise in pharmacology,
molecular biology, and medicinal chemistry.  You receive a collection of
PubMed abstracts and patent summaries and extract the most relevant
scientific intelligence for a drug discovery project.
"""

def literature_synthesis_prompt(query: str, abstracts: list[str]) -> list[dict]:
    combined = "\n\n---\n\n".join(
        [f"[{i+1}] {a}" for i, a in enumerate(abstracts[:15])]
    )
    user = f"""
RESEARCH QUERY: {query}

Below are retrieved literature abstracts. Analyse them and return a JSON
object (no markdown fences) with:
  - "key_targets": list of mentioned biological targets/proteins
  - "known_mechanisms": list of described mechanisms of action
  - "known_compounds": list of mentioned drug names or SMILES if available
  - "research_gaps": list of identified gaps or open questions
  - "summary": a 3-5 sentence synthesis of the state-of-the-art

ABSTRACTS:
{combined}
"""
    return build_messages(RETRIEVER_SYSTEM, user)


# ===========================================================================
# 3. HYPOTHESIS AGENT
# ===========================================================================

HYPOTHESIS_SYSTEM = """
You are a senior medicinal chemist and molecular biologist. You evaluate
evidence from literature and patents to select the most druggable target and
articulate a clear, testable mechanistic hypothesis for a new therapeutic.
"""

def hypothesis_prompt(
    disease: str,
    lit_findings: dict,
    patent_findings: dict,
    target_candidates: list[dict],
) -> list[dict]:
    user = f"""
DISEASE / INDICATION: {disease}

LITERATURE FINDINGS:
{lit_findings}

PATENT LANDSCAPE:
{patent_findings}

TARGET CANDIDATES (from UniProt/PDB lookup):
{target_candidates}

Based on all evidence above, return a JSON object (no markdown fences) with:
  - "selected_target": {{
        "gene_name": str,
        "uniprot_id": str,
        "pdb_id": str,
        "rationale": str   (2-3 sentences)
    }}
  - "hypothesis": str  (one clear mechanistic statement, ≤100 words)
  - "justification": str  (supporting evidence in 3-5 sentences)
  - "druggability_score": float between 0 and 1
  - "confidence": "high" | "medium" | "low"
"""
    return build_messages(HYPOTHESIS_SYSTEM, user)


# ===========================================================================
# 4. MOLECULE DESIGNER AGENT
# ===========================================================================

MOLECULE_DESIGNER_SYSTEM = """
You are a computational medicinal chemist specialising in structure-activity
relationships and de novo drug design. You design novel small molecules
targeting a given protein based on known pharmacophore features.
"""

def pharmacophore_prompt(
    target: dict,
    known_binders: list[str],
    pocket_description: str,
) -> list[dict]:
    binders_str = "\n".join(known_binders[:10]) if known_binders else "None available"
    user = f"""
TARGET: {target.get('gene_name', 'Unknown')} ({target.get('uniprot_id', '')})
BINDING POCKET DESCRIPTION: {pocket_description}

KNOWN BINDERS / REFERENCE SMILES:
{binders_str}

Extract the key pharmacophoric features (hydrogen bond donors/acceptors,
hydrophobic regions, aromatic rings, charged groups) and return a JSON object
(no markdown fences) with:
  - "pharmacophore_features": list of str
  - "core_scaffolds": list of SMILES strings to use as starting points
  - "forbidden_groups": list of PAINS/toxic substructures to avoid
  - "design_strategy": str (brief description of design rationale)
"""
    return build_messages(MOLECULE_DESIGNER_SYSTEM, user)


def molecule_refinement_prompt(
    smiles_list: list[str],
    target: dict,
    pharmacophore: dict,
) -> list[dict]:
    user = f"""
TARGET: {target.get('gene_name', 'Unknown')}
PHARMACOPHORE: {pharmacophore}

GENERATED SMILES CANDIDATES:
{chr(10).join(smiles_list)}

For each candidate, briefly comment on its predicted fit to the pharmacophore.
Then identify the top 5 most promising candidates based on drug-likeness and
target complementarity. Return a JSON object (no markdown fences) with:
  - "top_candidates": list of {{
        "smiles": str,
        "rationale": str,
        "predicted_activity": "high"|"medium"|"low"
    }}
  - "design_notes": str
"""
    return build_messages(MOLECULE_DESIGNER_SYSTEM, user)


# ===========================================================================
# 5. DOCKING EVALUATOR AGENT
# ===========================================================================

DOCKING_SYSTEM = """
You are a structural bioinformatician who interprets molecular docking results
and relates binding poses and energies to likely biological activity.
"""

def docking_interpretation_prompt(
    target: dict,
    docking_results: list[dict],
) -> list[dict]:
    results_str = "\n".join(
        [f"  {r['smiles'][:50]}... → {r['score']} kcal/mol" for r in docking_results]
    )
    user = f"""
TARGET: {target.get('gene_name', 'Unknown')} (PDB: {target.get('pdb_id', 'N/A')})

DOCKING RESULTS (AutoDock Vina binding energies):
{results_str}

Interpret these results. Rank the compounds from most to least promising.
Return a JSON object (no markdown fences) with:
  - "ranked_compounds": list of {{
        "smiles": str,
        "score_kcal_mol": float,
        "interpretation": str,
        "predicted_ic50_nm": str  (rough estimate or "N/A")
    }}
  - "best_candidate": str  (SMILES of top compound)
  - "structural_insights": str  (2-3 sentences on binding mode)
"""
    return build_messages(DOCKING_SYSTEM, user)


# ===========================================================================
# 6. SYNTHESIS EVALUATOR AGENT
# ===========================================================================

SYNTHESIS_SYSTEM = """
You are an expert synthetic organic chemist with experience in medicinal
chemistry campaigns. You propose practical, step-by-step retrosynthetic routes
for drug candidates, considering commercial availability of starting materials,
reagent costs, and reaction robustness.
"""

def retrosynthesis_prompt(smiles: str, sa_score: float, target: dict) -> list[dict]:
    user = f"""
DRUG CANDIDATE SMILES: {smiles}
SYNTHETIC ACCESSIBILITY SCORE: {sa_score:.2f} (1=easy, 10=hard)
TARGET: {target.get('gene_name', 'Unknown')}

Propose a retrosynthetic analysis and then a forward synthetic route for this
molecule. Return a JSON object (no markdown fences) with:
  - "retrosynthetic_steps": list of str  (disconnection steps, most complex → simple)
  - "forward_route": list of {{
        "step": int,
        "reaction": str,
        "reagents": str,
        "conditions": str,
        "expected_yield_percent": int
    }}
  - "starting_materials": list of str  (names or SMILES, commercially available)
  - "estimated_steps": int
  - "feasibility": "straightforward" | "moderate" | "challenging"
  - "key_challenges": list of str
"""
    return build_messages(SYNTHESIS_SYSTEM, user)


# ===========================================================================
# 7. REPORT COMPILER AGENT
# ===========================================================================

REPORT_SYSTEM = """
You are a scientific writer and drug discovery project lead. You compile
technical findings from a multi-stage AI pipeline into a polished, publication-
quality Project Proposal Report written in clear, professional English.
"""

def executive_summary_prompt(state: dict) -> list[dict]:
    user = f"""
Based on the following pipeline outputs, write a concise executive summary
(300-400 words) for a drug discovery project proposal:

DISEASE/TARGET INPUT: {state.get('input', '')}
SELECTED TARGET: {state.get('hypothesis', {}).get('selected_target', {})}
HYPOTHESIS: {state.get('hypothesis', {}).get('hypothesis', '')}
BEST CANDIDATE SMILES: {state.get('docking_results', [{}])[0].get('smiles', 'N/A') if state.get('docking_results') else 'N/A'}
TOP DOCKING SCORE: {state.get('docking_results', [{}])[0].get('score', 'N/A') if state.get('docking_results') else 'N/A'} kcal/mol

Write a compelling executive summary suitable for a pharmaceutical R&D audience.
Include: scientific rationale, target selection justification, key findings
from in silico studies, and next steps.
"""
    return build_messages(REPORT_SYSTEM, user)


def section_polish_prompt(section_title: str, raw_content: str) -> list[dict]:
    user = f"""
Polish the following draft section of a drug discovery project proposal.
Improve clarity, flow, and scientific precision without changing the facts.
Add appropriate transitions. Keep it concise (max 500 words).

SECTION: {section_title}

DRAFT:
{raw_content}

Return only the polished text – no JSON, no markdown code fences.
"""
    return build_messages(REPORT_SYSTEM, user)