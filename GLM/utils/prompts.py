# drug_discovery_pipeline/utils/prompts.py
"""
Prompt templates used by every agent in the pipeline.
Improved for precision, structure, and to eliminate empty / ambiguous responses.
"""

# ── Planner ───────────────────────────────────────────────────────────────────

PLAN_SYSTEM = (
    "You are a senior drug-discovery project manager. "
    "Given a disease indication or a biological target name, produce a concise "
    "execution plan with exactly six numbered phases. Each phase must have a one-line "
    "description. You MUST respond with the plan text and nothing else. "
    "If you truly cannot generate a plan, output 'Unable to generate plan.' "
    "Do NOT leave your response blank."
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
    "For each phase, write exactly one sentence that describes the specific action or "
    "question to be addressed (no generalities). Also state clearly whether the input "
    "appears to be a *disease* or a *target*.\n\n"
    "Format your response as:\n"
    "Type: disease / target\n"
    "1. <description>\n"
    "2. <description>\n"
    "... (up to 6)\n\n"
    "Respond NOW with the plan. Do not add any other text."
)

# ── Retriever: search-query generation ────────────────────────────────────────

SEARCH_QUERY_PROMPT = (
    "Given the following drug-discovery topic, generate exactly THREE specific PubMed "
    "search queries (each no more than 8 words) that will maximise relevant results about "
    "disease mechanisms, known drug targets, and small-molecule modulators. "
    "Use Boolean operators (AND/OR) where helpful. Do not use quotes.\n\n"
    "Topic: \"{input}\"\n\n"
    "Write three queries below, one per line, no numbering, no extra text:\n"
)

PATENT_QUERY_PROMPT = (
    "Given the following drug-discovery topic, generate exactly TWO specific patent "
    "search queries (each no more than 6 words) focused on small-molecule therapeutics. "
    "Use Boolean operators (AND/OR) where helpful. Do not use quotes.\n\n"
    "Topic: \"{input}\"\n\n"
    "Write two queries below, one per line, no numbering, no extra text:\n"
)

# ── Hypothesis ────────────────────────────────────────────────────────────────

HYPOTHESIS_SYSTEM = (
    "You are an expert molecular biologist and pharmacologist. "
    "Your task is to evaluate candidate drug targets, select the single best, and "
    "formulate a testable mechanistic hypothesis for therapeutic intervention. "
    "You MUST output exactly the requested format. If any field is unknown, use 'N/A'. "
    "Never leave the response empty."
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
    "1. Rank the provided candidate targets (in your mind only – do not output the ranking). "
    "Base your ranking on druggability, strength of evidence, and novelty. "
    "If no targets are supplied, propose the most plausible one yourself.\n"
    "2. Select the SINGLE best target.\n"
    "3. Formulate a mechanistic hypothesis (≤150 words) explaining HOW a small-molecule "
    "modulator of this target would treat the disease. Include the expected pharmacological "
    "effect (inhibition/activation) and downstream pathway impact.\n"
    "4. Provide a justification (≤100 words) for selecting this target over others.\n\n"
    "Respond EXACTLY in the following format (no markdown fences, no extra commentary):\n\n"
    "selected_target: <Gene Symbol>\n"
    "uniprot_id: <UniProt Accession or N/A>\n"
    "pdb_id: <PDB ID or N/A>\n"
    "hypothesis: |\n"
    "  <hypothesis text here>\n"
    "justification: |\n"
    "  <justification text here>\n\n"
    "Write your response NOW."
)

# ── Pharmacophore extraction ──────────────────────────────────────────────────

PHARMACOPHORE_PROMPT = (
    "Given the following target information and mechanistic hypothesis, "
    "list the key pharmacophore features a small-molecule ligand should possess "
    "to achieve potent and selective modulation.\n\n"
    "Target: {target} (UniProt: {uniprot_id})\n"
    "Hypothesis: {hypothesis}\n\n"
    "List 4-6 features as a bulleted list (using '- '). For each feature, specify "
    "the type of interaction (e.g., hydrogen-bond donor, hydrophobic, π-π stacking, "
    "ionic, halogen bond) and, if possible, the binding pocket region or residue it "
    "would target.\n\n"
    "Example:\n"
    "- Hydrogen-bond donor to interact with the hinge region (e.g., residue X)\n"
    "- Hydrophobic aromatic ring occupying the selectivity pocket\n\n"
    "Write the features NOW. Do not leave this blank."
)

# ── Retrosynthesis ────────────────────────────────────────────────────────────

RETROSYNTHESIS_SYSTEM = (
    "You are a senior medicinal chemist and retrosynthesis expert. "
    "Given a small-molecule SMILES, propose a plausible 3-4 step retrosynthetic "
    "analysis using commercially available building blocks and robust named reactions. "
    "You MUST output the route in the specified format. If the SMILES is invalid or "
    "a route cannot be proposed, state 'Synthetic route cannot be proposed.' "
    "Do not leave blank."
)

RETROSYNTHESIS_PROMPT = (
    "SMILES: {smiles}\n"
    "Target: {target}\n"
    "Hypothesis: {hypothesis}\n\n"
    "Propose a retrosynthetic route with 3-4 steps. For each step, provide:\n"
    "- The transformation (reaction name)\n"
    "- Starting materials (SMILES if possible)\n"
    "- Reagents and conditions\n\n"
    "Format your answer as a numbered list, then finish with an overall assessment.\n\n"
    "Example format:\n"
    "Step 1: [Reaction name]\n"
    "  Starting material(s): <SMILES or description>\n"
    "  Product: <SMILES or description>\n"
    "  Conditions: <reagents, solvent, temperature>\n"
    "...\n"
    "Synthetic Feasibility: Easy / Medium / Hard\n\n"
    "Write your answer NOW. Do not include any additional commentary."
)

# ── Report compilation ────────────────────────────────────────────────────────

REPORT_SYSTEM = (
    "You are a professional scientific writer specialising in drug-discovery project "
    "proposals. Produce a polished, well-structured Markdown report that integrates "
    "all provided data. You MUST write the full report with every requested section. "
    "If a section lacks data, write 'Insufficient data available.' but never omit a "
    "section. Do NOT leave the report empty."
)

REPORT_PROMPT = (
    "Compile the following information into a comprehensive Project Proposal Report "
    "in Markdown format.\n\n"
    "=== USER INPUT ===\n{input}\n\n"
    "=== LITERATURE FINDINGS ===\n{literature}\n\n"
    "=== PATENT FINDINGS ===\n{patents}\n\n"
    "=== TARGET SELECTION ===\n{target}\n\n"
    "=== HYPOTHESIS ===\n{hypothesis}\n\n"
    "=== PHARMACOPHORE ===\n{pharmacophore}\n\n"
    "=== DESIGNED MOLECULES ===\n{molecules}\n\n"
    "=== DOCKING RESULTS ===\n{docking}\n\n"
    "=== SYNTHETIC ANALYSIS ===\n{synthesis}\n\n"
    "=== INSTRUCTIONS ===\n"
    "The report must contain the following sections in order. Use proper Markdown "
    "headings (##) and subheadings where helpful. Use tables for numeric data (e.g., "
    "docking scores, molecular properties). Cite PMIDs / patent numbers inline as "
    "[PMID: xxx] or [Patent: xxx].\n\n"
    "1. Executive Summary (150-200 words synthesising all findings)\n"
    "2. Background & Rationale\n"
    "3. Literature & Patent Landscape\n"
    "4. Target Selection & Mechanistic Hypothesis\n"
    "5. Drug-Candidate Design (include pharmacophore, designed molecules)\n"
    "6. In-Silico Evaluation (docking results, predicted properties)\n"
    "7. Proposed Synthetic Route (summary of retrosynthesis)\n"
    "8. Risk Assessment & Mitigation\n"
    "9. Next Steps & Timeline (12-month plan with milestones)\n"
    "10. References\n\n"
    "Begin your response with the Executive Summary. "
    "Write the FULL report NOW."
)