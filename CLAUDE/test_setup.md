# First-time setup
./scripts/setup_conda_env.sh
conda activate drugdisco

# Fill in remote URL
nano .env   # set REMOTE_OLLAMA_URL=http://<actual-host>:11434/v1

# Pre-flight check
./scripts/check_environment.sh

# Run pipeline
make run QUERY="Alzheimer's disease"

# Or directly
``` python
python main.py "EGFR non-small cell lung cancer" --save-state
```

# Check servers
make check-servers

# Run tests
make test

# All dev tools
make check