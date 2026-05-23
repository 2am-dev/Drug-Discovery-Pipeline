# drug_discovery_pipeline/main.py
# =============================================================================
# FILE: main.py
# ROLE: Command-line entry point for the Drug Discovery Pipeline.
#       Parses user arguments, initialises logging, creates the PlannerAgent,
#       and prints a summary of results.
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path (important when running as a script)
sys.path.insert(0, str(Path(__file__).parent))

from utils.helpers import setup_logging
from agents.planner import PlannerAgent

log = setup_logging("drug_discovery")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End AI Drug Discovery Pipeline (Ollama-powered)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Alzheimer's disease"
  python main.py "EGFR-mutant non-small cell lung cancer"
  python main.py "BRAF V600E melanoma" --output custom_report.md
  python main.py "Type 2 diabetes" --save-state
        """,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Disease indication, biological target, or therapeutic area",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output report filename (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save the full pipeline state dict as JSON alongside the report",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model name (e.g. llama3:8b)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def print_summary(state: dict) -> None:
    """Print a brief terminal summary after the pipeline completes."""
    print("\n" + "=" * 70)
    print("  DRUG DISCOVERY PIPELINE — COMPLETED")
    print("=" * 70)
    print(f"  Query:      {state.get('input')}")
    print(f"  Run ID:     {state.get('run_id')}")
    print(f"  Started:    {state.get('started_at')}")
    print(f"  Completed:  {state.get('completed_at')}")

    hyp = state.get("hypothesis", {}).get("selected_target", {})
    print(f"\n  Target:     {hyp.get('gene_name', 'N/A')} "
          f"({hyp.get('uniprot_id', '')}) — PDB: {hyp.get('pdb_id', 'N/A')}")

    hypothesis_text = state.get("hypothesis", {}).get("hypothesis", "")
    if hypothesis_text:
        print(f"\n  Hypothesis: {hypothesis_text[:120]}...")

    dr = state.get("docking_results", [])
    if dr:
        print(f"\n  Best docking score: {dr[0].get('score')} kcal/mol")
        print(f"  Top SMILES:         {dr[0].get('smiles', '')[:70]}")

    synth = state.get("synthesis", [])
    if synth:
        print(f"\n  Synthesis feasibility (top hit): {synth[0].get('feasibility', 'N/A')}")
        print(f"  SA Score: {synth[0].get('sa_score', 'N/A')}")

    print(f"\n  Report saved: {state.get('report_path', 'N/A')}")

    errors = state.get("errors", [])
    if errors:
        print(f"\n  ⚠  {len(errors)} non-fatal error(s) during run (see log):")
        for e in errors:
            print(f"     - {e.get('agent')}: {str(e.get('error'))[:80]}")

    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()

    # Apply overrides from CLI
    import os
    os.environ["LOG_LEVEL"] = args.log_level
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    # Re-setup logging with correct level
    setup_logging("drug_discovery")
    log.info(f"Drug Discovery Pipeline starting — query: '{args.query}'")

    # Run the pipeline
    planner = PlannerAgent()
    state = planner.run(args.query)

    # Override report filename if requested
    if args.output and state.get("report"):
        from utils.helpers import save_text
        output_path = Path(args.output)
        save_text(output_path, state["report"])
        state["report_path"] = str(output_path)
        log.info(f"Report also saved to custom path: {output_path}")

    # Optionally save full state
    if args.save_state:
        from config import OUTPUT_DIR
        state_path = OUTPUT_DIR / f"state_{state.get('run_id', 'unknown')}.json"
        try:
            # Make state JSON-serialisable
            serialisable = json.loads(
                json.dumps(state, default=str)
            )
            state_path.write_text(
                json.dumps(serialisable, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info(f"Pipeline state saved: {state_path}")
        except Exception as e:
            log.warning(f"Could not save state JSON: {e}")

    print_summary(state)


if __name__ == "__main__":
    main()