# drug_discovery_pipeline/main.py
"""
Entry point for the Drug Discovery Hypothesis-to-Report Pipeline.

Usage:
    python main.py "Alzheimer's disease"
    python main.py "BACE1"
    python main.py "non-small cell lung cancer" --receptor path/to/receptor.pdbqt
"""

from __future__ import annotations

import argparse
import sys
import logging

from config import OUTPUT_DIR
from utils.helpers import setup_logging
from agents.planner import PlannerAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End Drug Discovery Hypothesis-to-Report Pipeline"
    )
    parser.add_argument(
        "input",
        type=str,
        help='Disease indication (e.g. "Alzheimer\'s disease") or biological target (e.g. "BACE1").',
    )
    parser.add_argument(
        "--receptor",
        type=str,
        default="",
        help="Path to a prepared receptor PDBQT file for docking (optional).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override config log level from CLI
    import config
    config.LOG_LEVEL = args.log_level
    setup_logging()

    logger = logging.getLogger("main")
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Initialise shared state
    state: dict = {
        "input": args.input,
        "input_type": "",           # filled by planner
        "receptor_pdbqt": args.receptor,
        # ── to be populated by agents ──
        "plan": "",
        "literature_results": [],
        "patent_results": [],
        "target_info": {},
        "hypothesis": {},
        "pharmacophore": "",
        "generated_molecules": [],
        "shortlisted_molecules": [],
        "docking_results": [],
        "synthesis_results": [],
        "report_md": "",
        "report_path": "",
    }

    # Run pipeline
    planner = PlannerAgent()
    try:
        state = planner.run(state)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed with error: %s", exc)
        sys.exit(1)

    # Print summary
    report_path = state.get("report_path", "")
    if report_path:
        logger.info("✅ Report saved to: %s", report_path)
        print(f"\n{'='*60}")
        print(f"  Report saved to: {report_path}")
        print(f"{'='*60}\n")
    else:
        logger.error("No report was generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()