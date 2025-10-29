# RAG Team Implementation Guide

## Structure

```
implementation/
├── answer.py          # Baseline implementation
├── ingest.py          # Baseline implementation
├── dkisselev-zz/      # Team member implementations
│   ├── answer.py
│   └── ingest.py
```

## Switching Implementations

Set the `RAG_IMPLEMENTATION` environment variable:

```bash
# Use baseline
export RAG_IMPLEMENTATION=baseline

# Use team member implementation
export RAG_IMPLEMENTATION=dkisselev-zz
```

Or inline:
```bash
RAG_IMPLEMENTATION=dkisselev-zz uv run app.py
```

## Running Evaluations

### Quick Evaluation (Shell Script)
```bash
./run_eval.sh baseline          # Evaluate baseline
./run_eval.sh dkisselev-zz      # Evaluate your implementation
```

### Full UI Evaluation
```bash
RAG_IMPLEMENTATION=baseline uv run evaluator.py
```

### Command Line Evaluation
```bash
RAG_IMPLEMENTATION=baseline uv run python -c "
from evaluation.eval import evaluate_all_retrieval
for test, result, _ in evaluate_all_retrieval():
    print(f'{result.mrr:.3f}', end=' ')
"
```

## Ingesting Data

```bash
# Ingest with baseline
cd implementation && uv run ingest.py

# Ingest with team implementation
cd implementation/dkisselev-zz && uv run ingest.py
```
