# Node.js Extraction Pipeline (EVO_34)

> Extracted from claude.md — Historical reference.

High-speed extraction of training data from Jupyter notebooks using Node.js for 10x faster JSON parsing and parallel regex extraction.

## Configuration

```yaml
extraction:
  engine: "Node.js (v24.11.1)"
  script: "extract_kernel_data.js"
  source: "advanced_kernel_research.ipynb"
  output: "kernel_extracted_data.jsonl"
  stats: "kernel_extraction_stats.json"
  target: "22+ Million Parameters"

status:
  last_run: "2026-01-24T04:25:00.000Z"
  total_examples: 1374
  vocabulary_size: 81047
  parameter_estimate: 7.1B
  coherence_score: 1.0
```

## Core Commands

```bash
node extract_kernel_data.js
python rebuild_kernel_complete.py
```
