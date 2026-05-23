# drug_discovery_pipeline/utils/prompts.py
"""
Prompt templates used by every agent in the pipeline.
Each constant / function returns a ready-to-use prompt string.
"""

# ── Planner ───────────────────────────────────────────────────────────────────

PLAN_SYSTEM = (
    "You are a senior drug-discovery project manager. "
    "Given a disease indication or a biological target name, produce a concise "
    "execution plan with numbered phases. Each phase should have a one-line "
    "description. Do NOT execute the plan – just return it as text."
)

PLAN_PROMPT = (
    "The user input is: \"{input}\"\n\n"
    "Decompose this into a drug-discovery pipeline plan with the following phases:\n"
    "1. Literature & Patent Mining\n"
    "2. Target Selection & Hypothesis Formulation\n"
    "3. In-silico Molecule Design\n"
    "4. Docking & Binding-Affinity Prediction\n"
    "5. Synthetic Accessibility & Route Proposal\n"
    "6. Report Compilation\n\n"
    "For each phase, write 1-2 sentences describing what will be done. "
    "Also state whether the input appears to be a *disease* or a *target*."
)

# ── Retriever: search-query generation ────────────────────────────────────────

SEARCH_QUERY_PROMPT = (
    "Given the following drug-discovery topic, generate THREE specific PubMed "
    "search queries (each ≤ 8 words) that will maximise relevant results about "
    "disease mechanisms, known drug targets, and small-molecule modulators.\n\n"
    "Topic: \"{input}\"\n\n"
    "Return ONLY the three queries, one per line, no numbering."
)

PATENT_QUERY_PROMPT = (
    "Given the following drug-discovery topic, generate TWO specific patent "
    "search queries (each ≤ 6 words) focused on small-molecule therapeutics.\n\n"
    "Topic: \"{input}\"\n\n"
    "Return ONLY the two queries, one per line, no numbering."
)

# ── Hypothesis ────────────────────────────────────────────────────────────────

HYPOTHESIS_SYSTEM = (
    "You are an expert molecular biologist and pharmacologist. "
    "Your task is to rank candidate drug targets and formulate a clear, "
    "testable mechanistic hypothesis for therapeutic intervention."
)

HYPOTHESIS_PROMPT = (
    "=== INPUT ===\n"
    "User query: {input}\n\n"
    "=== LITERATURE EVIDENCE ===\n"
    "{literature}\n\n"
    "=== PATENT EVIDENCE ===\n"
    "{patents}\n\n"
    "=== TARGET CANDIDATES ===\n"
    "{targets}\n\n"
    "=== INSTRUCTIONS ===\n"
    "1. Rank the candidate targets by druggability, evidence strength, and novelty.\n"
    "2. Select the SINGLE best target.\n"
    "3. Formulate a mechanistic hypothesis (≤ 150 words) explaining HOW a "
    "small-molecule modulator of this target would treat the disease.\n"
    "4. Provide a brief justification (≤ 100 words).\n\n"
    "Respond in the following YAML-like format (no markdown fences):\n\n"
    "selected_target: <Gene Symbol>\n"
    "uniprot_id: <UniProt Accession>\n"
    "pdb_id: <PDB ID or 'N/A'>\n"
    "hypothesis: |\n  <hypothesis text>\n"
    "justification: |\n  <justification text>\n"
)

# ── Pharmacophore extraction ──────────────────────────────────────────────────

PHARMACOPHORE_PROMPT = (
    "Given the following target information and mechanistic hypothesis, "
    "list the key pharmacophore features a small-molecule ligand should have.\n\n"
    "Target: {target} ({uniprot_id})\n"
    "Hypothesis: {hypothesis}\n\n"
    "List features as bullet points, e.g.:\n"
    "- Hydrogen-bond donor near <region>\n"
    "- Hydrophobic aromatic ring\n"
    "- Positive ionisable group\n"
    "- …\n\n"
    "Keep it to 4-6 features."
)

# ── Retrosynthesis ────────────────────────────────────────────────────────────

RETROSYNTHESIS_SYSTEM = (
    "You are a senior medicinal chemist and retrosynthesis expert. "
    "Given a small-molecule SMILES, propose a plausible 3-4 step "
    "retrosynthetic analysis using commercially available building blocks "
    "and well-known reactions."
)

RETROSYNTHESIS_PROMPT = (
    "SMILES: {smiles}\n"
    "Target: {target}\n"
    "Hypothesis: {hypothesis}\n\n"
    "Propose a retrosynthetic route with 3-4 steps. For each step provide:\n"
    "- The transformation (reaction name)\n"
    "- Starting materials (SMILES if possible)\n"
    "- Reagents / conditions\n\n"
    "End with an overall assessment of synthetic feasibility (Easy / Medium / Hard)."
)

# ── Report compilation ────────────────────────────────────────────────────────

REPORT_SYSTEM = (
    "You are a professional scientific writer specialising in drug-discovery "
    "project proposals. Produce a polished, well-structured Markdown report."
)

REPORT_PROMPT = (
    "Compile the following data into a comprehensive Project Proposal Report "
    "in Markdown format.\n\n"
    "=== USER INPUT ===\n{input}\n\n"
    "=== EXECUTIVE SUMMARY (write 150-200 words) ===\n\n"
    "=== LITERATURE FINDINGS ===\n{literature}\n\n"
    "=== PATENT FINDINGS ===\n{patents}\n\n"
    "=== TARGET SELECTION ===\n{target}\n\n"
    "=== HYPOTHESIS ===\n{hypothesis}\n\n"
    "=== PHARMACOPHORE ===\n{pharmacophore}\n\n"
    "=== DESIGNED MOLECULES ===\n{molecules}\n\n"
    "=== DOCKING RESULTS ===\n{docking}\n\n"
    "=== SYNTHETIC ANALYSIS ===\n{synthesis}\n\n"
    "=== INSTRUCTIONS ===\n"
    "Produce a report with these sections:\n"
    "1. Executive Summary\n"
    "2. Background & Rationale\n"
    "3. Literature & Patent Landscape\n"
    "4. Target Selection & Mechanistic Hypothesis\n"
    "5. Drug-Candidate Design\n"
    "6. In-Silico Evaluation (properties, docking, SA)\n"
    "7. Proposed Synthetic Route\n"
    "8. Risk Assessment & Mitigation\n"
    "9. Next Steps & Timeline (propose a 12-month plan)\n"
    "10. References\n\n"
    "Use proper Markdown headings (##), tables where appropriate, and cite "
    "PMIDs / patent numbers inline as [PMID: xxx] or [Patent: xxx]."
)