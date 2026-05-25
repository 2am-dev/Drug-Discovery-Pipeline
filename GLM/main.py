# drug_discovery_pipeline/main.py
"""
Entry point – dual-server, .env-driven.

Usage:
    python main.py "Alzheimer's disease"
    python main.py "BACE1" --receptor receptor.pdbqt
    python main.py "lung cancer" --remote http://192.168.1.50:11434/v1
"""

from __future__ import annotations

import argparse
import sys
import logging

import config
from config import OUTPUT_DIR
from utils.helpers import setup_logging, check_servers
from agents.planner import PlannerAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drug Discovery Hypothesis-to-Report Pipeline (dual-server)"
    )
    parser.add_argument(
        "input",
        type=str,
        help='Disease indication or biological target.',
    )
    parser.add_argument(
        "--receptor",
        type=str,
        default="",
        help="Path to a prepared receptor PDBQT file for docking.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=None,
        help="Override remote Ollama base URL (e.g. http://192.168.1.50:11434/v1).",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Override local Ollama base URL.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level (default: from .env or INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── CLI overrides (take precedence over .env) ────────────────────────
    if args.remote:
        config.REMOTE_OLLAMA_BASE_URL = args.remote
    if args.local:
        config.LOCAL_OLLAMA_BASE_URL = args.local
    if args.log_level:
        config.LOG_LEVEL = args.log_level

    setup_logging()

    logger = logging.getLogger("main")
    logger.info("Output directory : %s", OUTPUT_DIR)
    logger.info("Remote server     : %s  model: %s", config.REMOTE_OLLAMA_BASE_URL, config.REMOTE_LLM_MODEL)
    logger.info("Local  server     : %s  model: %s", config.LOCAL_OLLAMA_BASE_URL, config.LOCAL_LLM_MODEL)
    logger.info("Embedding endpoint: %s  model: %s", config.EMBEDDING_BASE_URL, config.EMBEDDING_MODEL)

    # ── Pre-flight health check ──────────────────────────────────────────
    logger.info("Running pre-flight health check…")
    health = check_servers()
    all_ok = True
    for srv in ("remote", "local", "embeddings"):
        info = health.get(srv, {})
        ok = info.get("ok", False)
        if not ok:
            logger.error("✗ %s – %s", srv, info.get("error", "unavailable"))
            all_ok = False
        else:
            logger.info("✓ %s – %s", srv, info.get("model", ""))

    if not all_ok:
        logger.error(
            "One or more servers unavailable. Check .env and that both Ollama instances are running.\n"
            "  Remote : %s\n"
            "  Local  : %s\n"
            "  Embed  : %s",
            config.REMOTE_OLLAMA_BASE_URL,
            config.LOCAL_OLLAMA_BASE_URL,
            config.EMBEDDING_BASE_URL,
        )
        logger.warning("Continuing anyway – calls will fail at runtime if servers stay down.")

    # ── Initialise shared state ──────────────────────────────────────────
    state: dict = {
        "input": args.input,
        "input_type": "",
        "receptor_pdbqt": args.receptor,
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

    # ── Run pipeline ─────────────────────────────────────────────────────
    planner = PlannerAgent()
    try:
        state = planner.run(state)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

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