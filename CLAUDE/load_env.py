# drug_discovery_pipeline/load_env.py
# =============================================================================
# FILE: load_env.py
# ROLE: Tiny bootstrap module — loads .env before any other import.
#       Import this at the very top of main.py or any entry-point script.
#
# Usage in main.py:
#   import load_env   # must be first import
#   from config import ...
# =============================================================================

from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv

    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)
        print(f"[load_env] Loaded: {env_file}")
    else:
        # Try explicit path relative to this file
        candidate = Path(__file__).parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            print(f"[load_env] Loaded: {candidate}")
        else:
            print("[load_env] No .env file found — using system environment only")

except ImportError:
    print(
        "[load_env] python-dotenv not installed. "
        "Run: pip install python-dotenv\n"
        "  or export variables manually before running the pipeline."
    )