```
cd drug_discovery_pipeline
```

# Basic run
``` python
python main.py "Alzheimer's disease"
```

# With custom output filename
``` python

python main.py "EGFR-mutant NSCLC" --output my_report.md
 ```
# Save full pipeline state as JSON
``` python

python main.py "Type 2 diabetes" --save-state
```
# Override LLM model
``` python

python main.py "BRAF V600E melanoma" --model llama3.1:8b
```
# Debug logging
``` python

python main.py "Parkinson's disease" --log-level DEBUG
```