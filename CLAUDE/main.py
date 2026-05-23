# drug_discovery_pipeline/main.py
# =============================================================================
# FILE: main.py
# ROLE: CLI entry point — unchanged in logic, adds server health summary.
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.helpers import setup_logging, server_is_healthy
from config import LOCAL_OLLAMA_URL, REMOTE_OLLAMA_URL, TASK_MODEL_MAP

log = setup_logging("drug_discovery")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Drug Discovery Pipeline — dual-server Ollama edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Alzheimer's disease"
  python main.py "EGFR NSCLC" --save-state
  python main.py "BRAF V600E melanoma" --model-override gemma4:31b-it-q8_0
  python main.py "Type 2 diabetes" --log-level DEBUG
        """,
    )
    p.add_argument("query", help="Disease / target / therapeutic area")
    p.add_argument("--output",         default=None, help="Custom report filename")
    p.add_argument("--save-state",     action="store_true", help="Save JSON state")
    p.add_argument("--model-override", default=None, help="Force a specific model")
    p.add_argument("--log-level",      default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def print_server_status() -> None:
    """Show which servers are reachable before the pipeline starts."""
    print("\n── Server Status ─────────────────────────────────────")
    for label, url in [("LOCAL  (5070 Ti)", LOCAL_OLLAMA_URL),
                        ("REMOTE (A6000x3)", REMOTE_OLLAMA_URL)]:
        ok = server_is_healthy(url)
        icon = "✓" if ok else "✗"
        print(f"  {icon} {label}  {url}")
    print("──────────────────────────────────────────────────────\n")


def print_routing_table() -> None:
    """Show the task→model→server routing table."""
    print("── Routing Table ─────────────────────────────────────")
    for task, (model, server) in TASK_MODEL_MAP.items():
        loc = "REMOTE" if server == REMOTE_OLLAMA_URL else "LOCAL "
        print(f"  {loc} | {model:<30} | {task}")
    print("──────────────────────────────────────────────────────\n")


def print_summary(state: dict) -> None:
    print("\n" + "=" * 70)
    print("  DRUG DISCOVERY PIPELINE — COMPLETED")
    print("=" * 70)
    print(f"  Query:     {state.get('input')}")
    print(f"  Run ID:    {state.get('run_id')}")
    print(f"  Started:   {state.get('started_at')}")
    print(f"  Completed: {state.get('completed_at')}")

    hyp = state.get("hypothesis", {}).get("selected_target", {})
    print(f"\n  Target:    {hyp.get('gene_name','N/A')} "
          f"({hyp.get('uniprot_id','')}) — PDB: {hyp.get('pdb_id','N/A')}")

    h = state.get("hypothesis", {}).get("hypothesis", "")
    if h:
        print(f"\n  Hypothesis: {h[:120]}...")

    dr = state.get("docking_results", [])
    if dr:
        print(f"\n  Best docking score: {dr[0].get('score')} kcal/mol")
        print(f"  Top SMILES:         {dr[0].get('smiles','')[:70]}")

    synth = state.get("synthesis", [])
    if synth:
        print(f"\n  Synthesis feasibility: {synth[0].get('feasibility','N/A')}")
        print(f"  SA Score:              {synth[0].get('sa_score','N/A')}")

    print(f"\n  Report: {state.get('report_path','N/A')}")

    errs = state.get("errors", [])
    if errs:
        print(f"\n  ⚠  {len(errs)} non-fatal error(s):")
        for e in errs:
            print(f"     - {e.get('agent')}: {str(e.get('error'))[:80]}")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()

    import os
    os.environ["LOG_LEVEL"] = args.log_level
    setup_logging("drug_discovery")

    print_server_status()
    print_routing_table()

    from agents.planner import PlannerAgent
    planner = PlannerAgent()
    state   = planner.run(args.query)

    if args.output and state.get("report"):
        from utils.helpers import save_text
        p = Path(args.output)
        save_text(p, state["report"])
        state["report_path"] = str(p)

    if args.save_state:
        from config import OUTPUT_DIR
        sp = OUTPUT_DIR / f"state_{state.get('run_id','?')}.json"
        try:
            sp.write_text(
                json.dumps(json.loads(json.dumps(state, default=str)), indent=2),
                encoding="utf-8",
            )
            log.info(f"State saved: {sp}")
        except Exception as e:
            log.warning(f"State save failed: {e}")

    print_summary(state)


if __name__ == "__main__":
    main()