#!/bin/bash
# Quick evaluation script for RAG implementations

IMPL=${1:-baseline}

echo "================================================"
echo "Running evaluation for: $IMPL"
echo "================================================"

export RAG_IMPLEMENTATION=$IMPL

# Run retrieval evaluation
echo ""
echo "RETRIEVAL EVALUATION"
echo "----------------------------------------"
uv run python -c "
import sys
from evaluation.eval import evaluate_all_retrieval

total_mrr = 0.0
total_ndcg = 0.0
total_coverage = 0.0
count = 0

for test, result, prog in evaluate_all_retrieval():
    count += 1
    total_mrr += result.mrr
    total_ndcg += result.ndcg
    total_coverage += result.keyword_coverage
    if count % 10 == 0:
        print(f'  Processed {count} tests...', end='\r', file=sys.stderr)

print(f'Completed {count} tests\n')
print(f'RETRIEVAL RESULTS:')
print(f'  MRR:      {total_mrr/count:.4f}')
print(f'  nDCG:     {total_ndcg/count:.4f}')
print(f'  Coverage: {total_coverage/count:.1f}%')
"

# Run answer evaluation
echo ""
echo ""
echo "ANSWER EVALUATION"
echo "----------------------------------------"
uv run python -c "
import sys
from evaluation.eval import evaluate_all_answers

total_accuracy = 0.0
total_completeness = 0.0
total_relevance = 0.0
count = 0

for test, result, prog in evaluate_all_answers():
    count += 1
    total_accuracy += result.accuracy
    total_completeness += result.completeness
    total_relevance += result.relevance
    if count % 10 == 0:
        print(f'  Processed {count} tests...', end='\r', file=sys.stderr)

print(f'Completed {count} tests\n')
print(f'ANSWER RESULTS:')
print(f'  Accuracy:     {total_accuracy/count:.2f}/5.00')
print(f'  Completeness: {total_completeness/count:.2f}/5.00')
print(f'  Relevance:    {total_relevance/count:.2f}/5.00')
"
