# dkisselev-zz RAG Implementation

Optimized RAG system achieving **+28% MRR improvement** through systematic testing and hyperparameter tuning.

## Key Changes & Techniques

### 1. GTE-small Embeddings
**Reason:** Baseline `all-MiniLM-L6-v2` embeddings were limiting retrieval quality  
**Technique:** Upgraded to `thenlper/gte-small` - embeddings from Alibaba with good semantic understanding  
**Result:** +21% MRR (0.72 → 0.91) - Single biggest improvement  
**Why it works:** Better trained on diverse text types, superior semantic understanding of business/contract language

### 2. Cross-Encoder Reranking
**Reason:** Initial retrieval can be noisy; need refinement  
**Technique:** Two-stage retrieval - fast semantic search (retrieve 20) → precise cross-encoder reranking (return 9)  
**Result:** +4.5% MRR improvement  
**Why it works:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) evaluates query-document pairs more precisely than bi-encoders

### 3. Optimized Chunking Strategy
**Reason:** Default 1000-char chunks fragmented documents too much  
**Technique:** Increased to 2000-char chunks with header-aware splitting (`## `, `### `)  
**Result:** +1.8% MRR, +7.2% completeness  
**Why it works:** Larger chunks preserve context, fewer fragments, complete information units. 

### 4. K=9 Document Selection
**Reason:** Finding optimal balance between coverage and noise  
**Technique:** Hyperparameter tuning tested K=3,4,5,6,7,8,9,10,12,15,20 with fine resolution  
**Result:** +0.2% MRR vs K=7, significantly better than K=3 or K=20  
**Why it works:** K=9 provides enough context without overwhelming the LLM.

### 5. Enhanced Metadata Extraction
**Reason:** Structured information in documents wasn't being captured  
**Technique:** Extract entity names, salaries, job titles, contract numbers, pricing from filenames and content using regex  
**Result:** Minimal retrieval impact, but improves answer formatting and context richness  
**Why it works:** LLM can reference structured metadata alongside content for more precise answers

### 6. Query Expansion
**Reason:** Hypothesis that multiple query variations improve coverage  
**Technique:** LLM generates 2 alternative phrasings, retrieve for each, combine results  
**Result:** +0.4% coverage BUT -1.7% accuracy - **Disabled**  
**Why it failed:** More queries = more noise.

### 7. Hybrid Search (BM25 + Semantic)
**Reason:** Hypothesis that combining keyword and semantic search improves accuracy  
**Technique:** BM25 keyword search + GTE-small semantic search, combine ~40 docs, rerank to top 9  
**Result:** +0.7% MRR BUT -2.8% completeness - **Disabled**  
**Why it failed:** BM25 adds keyword-focused but semantically weaker documents. BM25 diluted quality rather than improving it.

---

## Final Configuration

```python
# ingest.py
chunk_size = 2000
chunk_overlap = 400
separators = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# answer.py
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
RETRIEVAL_K = 20  # Retrieve candidates for reranking
FINAL_K = 9  # Return top 9 after cross-encoder reranking
USE_QUERY_EXPANSION = False  # Disabled (hurts accuracy)
USE_HYBRID_SEARCH = False  # Disabled (hurts completeness)
```

---

## Results vs Baseline

```
MRR:          0.7228 → 0.9280  (+28.4%)
nDCG:         0.7392 → 0.8994  (+21.7%)
Coverage:     80.8% → 97.1%    (+20.2%)
Accuracy:     4.00 → 4.69      (+17.3%)
Completeness: 3.35 → 3.98      (+18.8%)
Relevance:    4.72 → 4.94      (+4.7%)
```

**Production-ready:** Excellent retrieval (93% MRR), high answer quality (94% accuracy)

---

## Files

**Core Implementation:**
- `ingest.py` - Data ingestion with optimized chunking
- `answer.py` - Question answering with reranking


**Optimization:**
- `hyperparameter_tuning.py` - Automated tuning for chunk size, K, embeddings
- `hyperparameter_results.json` - Tuning results data
- `hyperparameter_*.png` - Visualization plots

**Analysis:**
- `analyze_knowledge_base.py` - Data analysis for chunking strategy
---

## Quick Commands

```bash
# Re-ingest data
uv run python ingest.py

# Test locally (from project root)
cd ../.. && RAG_IMPLEMENTATION=dkisselev-zz uv run app.py

# Run evaluation (from project root)
cd ../.. && ./run_eval.sh dkisselev-zz

# Hyperparameter tuning
uv run python hyperparameter_tuning.py
```

---

## Lessons Learned

1. **Embeddings matter most** - Single biggest improvement (+21% MRR)
2. **Quality > Quantity** - More documents doesn't mean better answers
3. **Bigger chunks work with better embeddings** - 2000 chars > 1500 > 1200 > 1000
4. **Reranking adds value** - Two-stage retrieval (fast → precise) is effective
5. **Simple is often better** - Pure semantic search beat complex multi-query approaches