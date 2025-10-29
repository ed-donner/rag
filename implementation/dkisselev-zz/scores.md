# RAG Improvement Tracking - dkisselev-zz

## Baseline (Original Implementation)

**Configuration:**
- RETRIEVAL_K: 3
- No metadata extraction
- No reranking
- No query expansion
- No hybrid search

**Results:**
```
 RETRIEVAL METRICS:
  MRR:         0.7228
  nDCG:        0.7392
  Coverage:    80.8%

 ANSWER METRICS):
  Accuracy:     4.00/5.00
  Completeness: 3.35/5.00
  Relevance:    4.72/5.00
```

---

## Enhanced Metadata

**Changes:**
- Added metadata extraction in `ingest.py`
- Extract entity names from filenames
- Parse structured data (salaries, contract numbers, pricing, job titles, locations)
- Entity-specific metadata for employees, contracts, products, company docs

**Results:**
```
RETRIEVAL METRICS:
  MRR:         0.7228  (Δ 0.0000, +0.0%)
  nDCG:        0.7392  (Δ 0.0000, +0.0%)
  Coverage:    80.8%   (Δ 0.0%, +0.0%)

ANSWER METRICS (LLM-as-Judge):
  Accuracy:     4.04/5.00  (Δ +0.04, +1.0%)
  Completeness: 3.37/5.00  (Δ +0.02, +0.6%)
  Relevance:    4.71/5.00  (Δ -0.01, -0.2%)
```

**Analysis:**
- Minimal improvement as expected - metadata alone doesn't help much with K=3
- Metadata is a foundation that will show benefits when combined with other improvements
- Slight accuracy improvement suggests better context quality

---

## Retrieve More (K=20) + Cross-Encoder Reranking

**Changes:**
- Increased RETRIEVAL_K from 3 → 20 (retrieve more candidates)
- Added cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Retrieve 20 docs, rerank with cross-encoder, return top 5 to LLM
- Two-stage retrieval: fast semantic search → precise reranking

**Results:**
```
RETRIEVAL METRICS:
  MRR:         0.7228  (Δ 0.0000, +0.0%)
  nDCG:        0.7392  (Δ 0.0000, +0.0%)
  Coverage:    80.8%   (Δ 0.0%, +0.0%)

ANSWER METRICS (LLM-as-Judge):
  Accuracy:     4.11/5.00  (Δ +0.11 from baseline, +2.8%)
  Completeness: 3.43/5.00  (Δ +0.08 from baseline, +2.4%)
  Relevance:    4.79/5.00  (Δ +0.07 from baseline, +1.5%)
```

**Analysis:**
- Retrieval metrics unchanged (they measure keyword presence, not ranking quality)
- **Answer quality improved** - reranking selects better documents
- Accuracy +2.8%, Completeness +2.4%, Relevance +1.5%
- Modest gains - need more aggressive techniques for bigger improvements

---

## K=10 docs + Metadata in Prompt

**Changes:**
- Increased FINAL_K from 5 → 10 (double the context to LLM)
- Added `format_doc_with_metadata()` to explicitly pass metadata to LLM
- Metadata now visible in structured format (Entity, Type, Salary, Job Title, etc.)
- LLM now sees rich context with metadata fields

**Results:**
```
RETRIEVAL METRICS:
  MRR:         0.7228  (Δ 0.0000, +0.0%)
  nDCG:        0.7392  (Δ 0.0000, +0.0%)
  Coverage:    80.8%   (Δ 0.0%, +0.0%)

ANSWER METRICS (LLM-as-Judge):
  Accuracy:     4.01/5.00  (Δ +0.01 from baseline, +0.2%) [DOWN from 4.11!]
  Completeness: 3.47/5.00  (Δ +0.12 from baseline, +3.6%) [Best so far]
  Relevance:    4.72/5.00  (Δ 0.00 from baseline, 0.0%) [DOWN from 4.79]
```

**Analysis:**
- ⚠️ **Mixed results** - Not clearly better than Step 2
- ✓ **Completeness improved** (+3.6% vs baseline) - more context helps
- ✗ **Accuracy decreased** (4.11 → 4.01) - too much context adds noise
- ✗ **Relevance unchanged** - extra docs dilute focus

**Key Insight:**
**More is not always better!** Adding 10 docs instead of 5:
- Helps completeness (more information available)
- Hurts accuracy (too much noise, harder to find exact answer)
- Hurts relevance (LLM distracted by extra context)

**The "Goldilocks" problem**: 
- K=3 (baseline): Too little context → Low completeness
- K=5 (Step 2): Better balance
- K=10 (Step 2B): Too much context → Accuracy drops

**Cumulative improvement from baseline:**
- Accuracy: +0.2% (worse than Step 2's +2.8%)
- Completeness: +3.6% (better than Step 2's +2.4%)
- Relevance: 0.0% (worse than Step 2's +1.5%)

**Recommendation:** Step 2 (K=5) was better overall than Step 2B (K=10)

---

## Step 2C: Optimized Chunking + K=7 ⭐ BEST SO FAR

**Changes:**
- Analyzed knowledge base: avg file size 4004 chars
- Increased chunk_size: 1000 → 1500 chars
- Increased chunk_overlap: 200 → 300 chars
- Added header-aware splitting (prefer breaking on `## ` and `### `)
- Result: 308 chunks (down from 413) - larger, more coherent
- Set FINAL_K = 7 (middle ground between 5 and 10)

**Results:**
```
🔍 RETRIEVAL METRICS:  ✓✓✓ SIGNIFICANT IMPROVEMENT!
  MRR:         0.7517  (Δ +0.0289 from baseline, +4.0%)
  nDCG:        0.7617  (Δ +0.0225 from baseline, +3.0%)
  Coverage:    82.1%   (Δ +1.3%, +1.6%)

💬 ANSWER METRICS (LLM-as-Judge):
  Accuracy:     3.91/5.00  (Δ -0.09 from baseline, -2.2%)
  Completeness: 3.59/5.00  (Δ +0.24 from baseline, +7.2%) ⭐ BEST!
  Relevance:    4.75/5.00  (Δ +0.03 from baseline, +0.6%)
```

**Analysis:**
- ✅ **CHUNKING OPTIMIZATION WORKED!**
- ✅ MRR +4.0% - Better retrieval quality
- ✅ nDCG +3.0% - Better ranking
- ✅ Coverage +1.6% - Finding more keywords
- ✅ Completeness +7.2% - **Best result so far!** More context per chunk
- ⚠️ Accuracy -2.2% - Slight drop (larger chunks = slightly noisier)
- ✓ Relevance stable - K=7 is good middle ground

**Key Insights:**

1. **Chunking matters MORE than K!**
   - Step 2 (K=5, 1000 chunks): Accuracy +2.8%, Completeness +2.4%
   - Step 2C (K=7, 1500 chunks): Accuracy -2.2%, Completeness +7.2%
   - Larger chunks preserve context → Better completeness

2. **Why retrieval improved:**
   - Fewer, larger chunks (413 → 308) means less fragmentation
   - Each chunk contains more complete information
   - Header-aware splitting keeps related content together

3. **Why accuracy dropped slightly:**
   - Larger chunks have more content
   - Slightly harder for LLM to find exact answer
   - Trade-off: completeness vs. precision

4. **K=7 is a good balance:**
   - Better than K=10 (too noisy)
   - More complete than K=5
   - Gives LLM enough context without overwhelming

**Cumulative improvement from baseline:**
- MRR: +4.0% ⭐ (vs +0% in Steps 1-2B)
- Accuracy: -2.2% (trade-off for completeness)
- Completeness: +7.2% ⭐ BEST (vs +3.6% in Step 2B)
- Relevance: +0.6%

**This is our best configuration so far!**

---

## Step 2D-Test1: GTE-small Embeddings 🚀 BREAKTHROUGH!

**Changes (Option B - Test embeddings only):**
- Changed embeddings: `all-MiniLM-L6-v2` → `thenlper/gte-small`
- Kept everything else from Step 2C: chunk_size=1500, FINAL_K=7, RETRIEVAL_K=20
- Regenerated vector database with new embeddings

**Results:**
```
🔍 RETRIEVAL METRICS:  ⭐⭐⭐ MASSIVE IMPROVEMENT!
  MRR:         0.9113  (Δ +0.1596 from Step 2C, +21.2%)
  nDCG:        0.8918  (Δ +0.1301 from Step 2C, +17.1%)
  Coverage:    95.9%   (Δ +13.8%, +16.8%)

💬 ANSWER METRICS (LLM-as-Judge):
  Accuracy:     4.61/5.00  (Δ +0.70 from Step 2C, +17.9%)
  Completeness: 3.82/5.00  (Δ +0.23 from Step 2C, +6.4%)
  Relevance:    4.96/5.00  (Δ +0.21 from Step 2C, +4.4%)
```

**Analysis:**
- 🔥 **GAME CHANGER!** Single biggest improvement so far
- ✅ **MRR jumped from 0.75 → 0.91** (+21% relative improvement!)
- ✅ **nDCG improved 0.76 → 0.89** (+17% improvement)
- ✅ **Coverage 82% → 96%** - Finding keywords almost everywhere now
- ✅ **All answer metrics improved significantly**
- ✅ **Accuracy now 4.61/5.00** - Highest yet!
- ✅ **Completeness 3.82/5.00** - Better than Step 2C
- ✅ **Relevance 4.96/5.00** - Nearly perfect!

**Key Insights:**

1. **Embeddings matter MORE than everything else combined!**
   - All previous steps (1, 2, 2B, 2C): ~4-7% total improvement
   - Just changing embeddings: +21% improvement
   - **This is the most impactful single change**

2. **Why GTE-small is so much better:**
   - State-of-the-art model from Alibaba
   - Better semantic understanding of business/contract language
   - Same 384 dimensions as MiniLM-L6 (no storage penalty)
   - Training data likely includes more diverse text types

3. **Retrieval quality now excellent:**
   - MRR 0.91 means we find the right doc on first try 91% of time
   - Coverage 95.9% means we're finding nearly all keywords
   - This gives the LLM much better context to work with

4. **Answer quality benefits from better retrieval:**
   - When LLM gets the right documents, it answers correctly
   - Accuracy jumped to 4.61/5.00 (vs 3.91 in Step 2C)
   - Completeness improved to 3.82 (vs 3.59 in Step 2C)
   - Relevance nearly perfect at 4.96

**Cumulative improvement from baseline:**
- MRR: +26.1% (0.7228 → 0.9113) ⭐⭐⭐
- Accuracy: +15.3% (4.00 → 4.61) ⭐⭐⭐
- Completeness: +14.0% (3.35 → 3.82) ⭐⭐⭐
- Relevance: +5.1% (4.72 → 4.96) ⭐⭐

**Validation of hyperparameter tuning:**
- Tuning predicted +12% MRR improvement (on 30 questions)
- Full evaluation shows +21% MRR improvement (on 150 questions)
- **Even better than predicted!**

**Next Steps (Option B - Continue testing individually):**
- ✅ Test 2D-Test2: Chunk size optimization (1500 → 1200)
- ✅ Test 2D-Test3: K value optimization (testing K=20 without reranking)
- Next: Option A - Optimal configuration
- Then: Option C - Steps 3-5 (Query Expansion, Hybrid Search)

---

## Step 2D-Test2: Chunk Size Optimization (1500 → 1200)

**Changes (Option B - Test chunk size):**
- Changed chunk_size: 1500 → 1200 (as recommended by hyperparameter tuning)
- Changed chunk_overlap: 300 → 240 (20% of chunk_size)
- Kept: GTE-small embeddings, FINAL_K=7, RETRIEVAL_K=20
- Result: 383 chunks (vs 308 with chunk_size=1500)

**Results:**
```
🔍 RETRIEVAL METRICS:  ✓ Slight improvement
  MRR:         0.9212  (Δ +0.0099 from Test1, +1.1%)
  nDCG:        0.8964  (Δ +0.0046 from Test1, +0.5%)
  Coverage:    96.3%   (Δ +0.4%, +0.4%)

💬 ANSWER METRICS (LLM-as-Judge):  ⚠️ Slight degradation
  Accuracy:     4.51/5.00  (Δ -0.10 from Test1, -2.2%)
  Completeness: 3.77/5.00  (Δ -0.05 from Test1, -1.3%)
  Relevance:    4.95/5.00  (Δ -0.01 from Test1, -0.2%)
```

**Analysis:**
- ✅ **Retrieval improved marginally:** More chunks (383 vs 308) = better granularity
- ⚠️ **Answer quality decreased marginally:** Smaller chunks = less context per chunk
- 🤔 **Unexpected finding:** Hyperparameter tuning (with MiniLM-L6) suggested 1200 was optimal
- 💡 **Reality check:** With GTE-small, chunk_size=1500 is actually slightly better

**Key Insights:**

1. **Embedding model affects optimal chunk size:**
   - MiniLM-L6 (tuning): Best at chunk_size=1200
   - GTE-small (full eval): Best at chunk_size=1500
   - Better embeddings can handle larger chunks more effectively

2. **Trade-off remains:**
   - Smaller chunks (1200): Better retrieval precision, less context
   - Larger chunks (1500): Slightly worse retrieval, better context for LLM
   - With excellent embeddings (GTE-small), context matters more

3. **Marginal difference:**
   - MRR difference: 0.9212 vs 0.9113 = ~1% (negligible)
   - Accuracy difference: 4.51 vs 4.61 = ~2% (noticeable but small)
   - **Could use either size comfortably**

**Recommendation:**
- **Keep chunk_size=1500** - Better answer quality (accuracy 4.61 vs 4.51)
- The retrieval improvement at 1200 (+1%) doesn't offset answer degradation (-2%)
- With GTE-small's superior semantic understanding, larger chunks work better

**Cumulative improvement from baseline:**
- MRR: +27.5% (0.7228 → 0.9212)
- Accuracy: +12.8% (4.00 → 4.51)
- Completeness: +12.5% (3.35 → 3.77)
- Relevance: +4.9% (4.72 → 4.95)

**Decision:** Reverting to chunk_size=1500 for next test.

---

## Step 2D-Test3: K=20 Without Reranking ⚠️

**Changes (Option B - Test removing reranking):**
- Removed cross-encoder reranking step
- Changed FINAL_K from 7 → 20 (return all retrieved documents)
- Hypothesis: GTE-small embeddings good enough that reranking not needed
- Kept: GTE-small embeddings, chunk_size=1500

**Results:**
```
🔍 RETRIEVAL METRICS:  ✗ Significant degradation
  MRR:         0.8700  (Δ -0.0413 from Test1, -4.5%)
  nDCG:        0.8480  (Δ -0.0438 from Test1, -4.9%)
  Coverage:    96.6%   (Δ +0.7%, +0.7%)

💬 ANSWER METRICS:  ⚠️ Rate limited (too much context!)
  ERROR: OpenAI rate limit - 20 docs create ~6000+ tokens per question
  Unable to complete evaluation - context too large
```

**Analysis:**
- ✗ **Reranking IS valuable** - Even with excellent GTE-small embeddings
- ✗ **K=20 hurts retrieval quality** - MRR dropped 4.5%
- ✗ **Too much context for LLM** - 20 documents = token limit issues
- ✓ **Coverage improved slightly** (+0.7%) but not worth the trade-offs

**Key Insights:**

1. **Reranking provides significant value (+4.5% MRR):**
   - GTE-small gets us close to relevant docs
   - Cross-encoder reranking fine-tunes the order
   - The two-stage approach (retrieve 20 → rerank → top 7) works well

2. **More documents != better results:**
   - K=7: Focused, high-quality context
   - K=20: Noisy, overwhelming for LLM
   - Quality > quantity for context

3. **GTE-small + reranking is synergistic:**
   - GTE-small: Great initial retrieval
   - Reranking: Polish to perfection
   - Both are needed for best results

4. **Practical limits:**
   - 20 documents hit token limits
   - Even without limits, LLM struggles with too much text
   - "Lost in the middle" problem - LLM attention degrades with long context

**Recommendation:**
- **Keep reranking** - Provides 4.5% MRR boost
- **Keep K=7** - Optimal balance for LLM comprehension
- **Keep two-stage retrieval** - Retrieve 20, rerank, return top 7

**Cumulative improvement from baseline (Test1 remains best):**
- MRR: +26.1% (0.7228 → 0.9113)
- Accuracy: +15.3% (4.00 → 4.61)  
- Completeness: +14.0% (3.35 → 3.82)
- Relevance: +5.1% (4.72 → 4.96)

**Decision:** Reverting to K=7 with reranking (Test1 configuration).

---

## Step 2D-Test4: K=9 (vs K=7)

**Changes:**
- Changed FINAL_K from 7 → 9 (per refined hyperparameter tuning)
- Kept: GTE-small, chunk_size=1500, reranking

**Results:**
```
🔍 RETRIEVAL METRICS:  ≈ Marginal improvement
  MRR:         0.9115  (Δ +0.0002 from Test1, +0.02%)
  nDCG:        0.8832  (Δ -0.0086 from Test1, -0.96%)
  Coverage:    96.1%   (Δ +0.2%, +0.2%)

💬 ANSWER METRICS:  Mixed results
  Accuracy:     4.58/5.00  (Δ -0.03 from Test1, -0.65%)
  Completeness: 3.97/5.00  (Δ +0.15 from Test1, +3.9%) ✓
  Relevance:    4.97/5.00  (Δ +0.01 from Test1, +0.2%)
```

**Analysis:**
- Completeness improved (+3.9%) but accuracy decreased slightly
- Extra 2 documents add information but also noise
- Need to combine with chunk optimization

---

## Step 2D-Test5: chunk_size=2000 + K=9 🏆 BEST EVER!

**Changes:**
- Changed chunk_size: 1500 → 2000 (per refined hyperparameter tuning)
- Changed chunk_overlap: 300 → 400 (20% of chunk_size)
- Kept FINAL_K: 9
- Result: 229 chunks (vs 308 with chunk_size=1500)

**Results:**
```
🔍 RETRIEVAL METRICS:  ⭐⭐⭐ BEST EVER!
  MRR:         0.9280  (Δ +0.0167 from Test1, +1.8%)
  nDCG:        0.8994  (Δ +0.0076 from Test1, +0.9%)
  Coverage:    97.1%   (Δ +1.2%, +1.3%)

💬 ANSWER METRICS:  ⭐⭐⭐ BEST EVER!
  Accuracy:     4.69/5.00  (Δ +0.08 from Test1, +1.7%)
  Completeness: 3.98/5.00  (Δ +0.16 from Test1, +4.2%)
  Relevance:    4.94/5.00  (Δ -0.02 from Test1, -0.4%)
```

**Analysis:**
- 🎉 **BREAKTHROUGH!** Best configuration across nearly all metrics
- ✅ **All retrieval metrics improved** - MRR, nDCG, Coverage all at peak
- ✅ **Accuracy highest ever** - 4.69/5.00 (better than all previous tests)
- ✅ **Completeness highest ever** - 3.98/5.00 (+18.8% from baseline!)
- ✅ **Relevance still excellent** - 4.94/5.00 (marginal decrease)

**Key Insights:**

1. **Larger chunks work better with GTE-small:**
   - 2000 chars = Fewer, richer chunks (229 vs 308)
   - Less fragmentation = Better context preservation
   - Files break into 1-2 pieces instead of 2-3
   - GTE-small can understand larger semantic units

2. **K=9 + chunk=2000 are synergistic:**
   - K=9 alone: Mixed results (Test4)
   - chunk=2000 alone: Not tested separately
   - **Combined: Excellent results** (Test5)
   - More context per chunk + more chunks to LLM = optimal

3. **Why accuracy improved:**
   - Larger chunks contain more complete information
   - K=9 provides sufficient context
   - Less fragmentation = easier for LLM to find answers

**Cumulative improvement from baseline:**
- MRR: +28.4% (0.7228 → 0.9280) ⭐⭐⭐
- Accuracy: +17.3% (4.00 → 4.69) ⭐⭐⭐
- Completeness: +18.8% (3.35 → 3.98) ⭐⭐⭐
- Relevance: +4.7% (4.72 → 4.94) ⭐⭐
- Coverage: +20.2% (80.8% → 97.1%) ⭐⭐⭐

**This is our best configuration!**

---

## Option A: FINAL OPTIMAL CONFIGURATION 🏆

**Based on comprehensive testing (Option B + refinements):**

```python
# ingest.py
chunk_size = 2000        # ← Updated from 1500
chunk_overlap = 400      # 20% of chunk_size
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# answer.py
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
RETRIEVAL_K = 20  # Retrieve candidates
FINAL_K = 9       # ← Updated from 7
# Keep cross-encoder reranking
```

**Results (Test 2D-Test5 = Final Optimal):**
- MRR: 0.9280 (+28.4% from baseline) ⭐⭐⭐
- Accuracy: 4.69/5.00 (+17.3% from baseline) ⭐⭐⭐
- Completeness: 3.98/5.00 (+18.8% from baseline) ⭐⭐⭐
- Relevance: 4.94/5.00 (+4.7% from baseline) ⭐⭐

**Key Findings from all testing:**
1. ✅ **GTE-small embeddings: +21% MRR** (biggest single win) ⭐⭐⭐⭐⭐
2. ✅ **chunk_size=2000: +1.8% MRR** (better than 1500) ⭐⭐⭐
3. ✅ **K=9 with reranking: +0.2% MRR** (better than K=7) ⭐⭐
4. ✅ **Reranking: +4.5% MRR** (vs no reranking) ⭐⭐⭐
5. ⭐ **Metadata: Minimal retrieval impact** but helps answer quality

**Relative importance of changes:**
1. Embeddings (GTE-small): ⭐⭐⭐⭐⭐ (+21% MRR) - CRITICAL
2. Reranking: ⭐⭐⭐ (+4.5% MRR) - Important
3. Chunk optimization (2000): ⭐⭐⭐ (+1.8% MRR) - Important
4. K value (9): ⭐⭐ (+0.2% MRR) - Helpful
5. Metadata: ⭐ (Minimal retrieval impact, helps answers)

---

## Alternative Embedding Models (Tested!)

Hyperparameter tuning tested 4 models:
1. ✅ **thenlper/gte-small** - **WINNER!** MRR 0.88 (+12% vs MiniLM-L6)
2. **all-MiniLM-L12-v2** - MRR 0.85 (+7.8%, good fallback)
3. **all-mpnet-base-v2** - MRR 0.80 (+1.3%, too slow)
4. **all-MiniLM-L6-v2** - MRR 0.79 (baseline)

---

## Step 3: Query Expansion ⚠️ Mixed Results

**Goal:** Use LLM to generate multiple query variations to improve retrieval coverage

**Implementation:**
- Use ChatGPT to rephrase user queries into 2-3 variations
- Retrieve documents for each variation (3 queries × 20 docs)
- Combine and deduplicate results
- Rerank all collected documents with cross-encoder
- Return top K=9

**Baseline (Test5 - No Expansion):**
- MRR: 0.9280
- Accuracy: 4.69/5.00
- Completeness: 3.98/5.00
- Relevance: 4.94/5.00

**Results (Step 3 - With Query Expansion):**
```
🔍 RETRIEVAL METRICS:  Mixed
  MRR:         0.9278  (Δ -0.0002 from Test5, -0.02%)
  nDCG:        0.9029  (Δ +0.0035 from Test5, +0.39%) ✓
  Coverage:    97.5%   (Δ +0.4%, +0.4%) ✓

💬 ANSWER METRICS:  Mixed
  Accuracy:     4.61/5.00  (Δ -0.08 from Test5, -1.7%) ✗
  Completeness: 4.00/5.00  (Δ +0.02 from Test5, +0.5%) ✓
  Relevance:    4.93/5.00  (Δ -0.01 from Test5, -0.2%)
```

**Analysis:**
- ⚠️ **Trade-off detected:** Better coverage but worse accuracy
- ✅ Coverage improved (+0.4%) - Query variations find more keywords
- ✅ nDCG improved (+0.39%) - Better ranking quality overall
- ✅ Completeness improved (+0.5%) - More complete information retrieved
- ✗ **Accuracy decreased (-1.7%)** - More documents = more noise
- ≈ MRR/Relevance roughly unchanged

**Why the accuracy drop?**
1. Query expansion retrieves ~60 candidate documents (3 queries × 20)
2. Even after reranking to top 9, some less relevant docs slip through
3. More context with slight quality dilution hurts precise answers
4. Good for "completeness" questions, bad for "factual accuracy" questions

**Trade-off assessment:**
- If **completeness** is priority: ✅ Keep query expansion
- If **accuracy** is priority: ✗ Disable query expansion
- Current config: Completeness 4.00 (excellent) vs Accuracy 4.61 (very good)

**Recommendation:** 
**Disable query expansion** - The -1.7% accuracy loss outweighs +0.5% completeness gain. Accuracy is typically more important than completeness for factual Q&A.

**Alternative approaches to try:**
1. Reduce number of query variations (1 variation instead of 2)
2. Increase FINAL_K to handle more candidates better
3. Use query expansion only for "complex" questions
4. Try different expansion prompt (more focused)

---

## Step 4: Hybrid Search (BM25 + Semantic) ⚠️ Mixed Results

**Goal:** Combine keyword-based BM25 search with semantic vector search

**Implementation:**
- BM25 keyword search retrieves top 20 documents
- GTE-small semantic search retrieves top 20 documents
- Combine and deduplicate (~40 unique docs)
- Rerank all with cross-encoder
- Return top K=9

**Baseline (Test5 - Semantic Only):**
- MRR: 0.9280
- Accuracy: 4.69/5.00
- Completeness: 3.98/5.00
- Relevance: 4.94/5.00

**Results (Step 4 - Hybrid BM25 + Semantic):**
```
🔍 RETRIEVAL METRICS:  ✓ Improved
  MRR:         0.9348  (Δ +0.0068 from Test5, +0.73%) ✓
  nDCG:        0.9053  (Δ +0.0059 from Test5, +0.66%) ✓
  Coverage:    97.7%   (Δ +0.6%, +0.6%) ✓

💬 ANSWER METRICS:  ✗ Degraded
  Accuracy:     4.62/5.00  (Δ -0.07 from Test5, -1.5%) ✗
  Completeness: 3.87/5.00  (Δ -0.11 from Test5, -2.8%) ✗
  Relevance:    4.96/5.00  (Δ +0.02 from Test5, +0.4%)
```

**Analysis:**
- ⚠️ **Similar pattern to Step 3:** Better retrieval, worse answers
- ✅ All retrieval metrics improved - MRR, nDCG, Coverage all up
- ✗ **Answer quality degraded** - Both accuracy and completeness decreased
- ✗ **Completeness hit hardest** (-2.8%) - Unexpected!

**Why hybrid search hurt performance?**
1. **BM25 adds keyword-focused but semantically weaker documents:**
   - BM25 favors exact keyword matches
   - May retrieve docs with keywords in wrong context
   - Dilutes the high-quality semantic results

2. **GTE-small embeddings are already excellent:**
   - Semantic search alone achieves MRR 0.93
   - Hard to improve on top-tier embeddings
   - BM25 adds noise more than signal

3. **More candidates = lower quality:**
   - 40 combined docs → rerank → top 9
   - Even with reranking, some lower-quality docs slip in
   - Quality > quantity for answer generation

**Trade-off assessment:**
- Retrieval improved: +0.73% MRR
- Answer quality decreased: -1.5% accuracy, -2.8% completeness
- **Not worth it** - Answer quality is more important

**Recommendation:**
**Disable hybrid search** - The answer quality degradation (-1.5% accuracy, -2.8% completeness) outweighs retrieval improvements (+0.73% MRR).

**Why it didn't work here:**
- Our semantic search (GTE-small) is already near-perfect
- Hybrid search typically helps when semantic search is weak
- In our case, adding BM25 adds noise, not signal

---

---

## 🏆 FINAL OPTIMAL CONFIGURATION & SUMMARY

### **Complete Journey:**

| Step | Change | MRR | Accuracy | Status |
|------|--------|-----|----------|--------|
| **Baseline** | MiniLM-L6, 1000 chunks, K=3 | 0.7228 | 4.00 | Starting point |
| Step 1 | + Metadata | 0.7228 | 4.04 | +1.0% accuracy |
| Step 2 | + Reranking, K=5 | 0.7228 | 4.11 | +2.8% accuracy |
| Step 2C | + Chunking 1500 | 0.7517 | 3.91 | +4.0% MRR |
| **Test1** | + GTE-small | 0.9113 | 4.61 | +21% MRR ⭐ |
| Test2 | + Chunk 1200 | 0.9212 | 4.51 | -2.2% accuracy |
| **Test5** | + Chunk 2000, K=9 | **0.9280** | **4.69** | **BEST** 🏆 |
| Step 3 | + Query expansion | 0.9278 | 4.61 | -1.7% accuracy ✗ |
| Step 4 | + Hybrid search | 0.9348 | 4.62 | -2.8% completeness ✗ |

### **Final Configuration (Test5):**

```python
# implementation/dkisselev-zz/ingest.py
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
chunk_size = 2000
chunk_overlap = 400  # 20% of chunk_size
separators = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]  # Header-aware

# implementation/dkisselev-zz/answer.py
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
RETRIEVAL_K = 20  # Retrieve candidates
FINAL_K = 9  # Return after reranking
USE_QUERY_EXPANSION = False  # Disabled (hurts accuracy)
USE_HYBRID_SEARCH = False  # Disabled (hurts completeness)
# Cross-encoder reranking: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```

### **Final Results:**

```
🔍 RETRIEVAL METRICS:
  MRR:         0.9280  (+28.4% from baseline)
  nDCG:        0.8994  (+21.7% from baseline)
  Coverage:    97.1%   (+20.2% from baseline)

💬 ANSWER METRICS:
  Accuracy:     4.69/5.00  (+17.3% from baseline)
  Completeness: 3.98/5.00  (+18.8% from baseline)
  Relevance:    4.94/5.00  (+4.7% from baseline)
```

### **Key Success Factors (Ranked by Impact):**

1. **⭐⭐⭐⭐⭐ GTE-small embeddings (+21% MRR)** - Game-changing upgrade
2. **⭐⭐⭐ Cross-encoder reranking (+4.5% MRR)** - Essential refinement
3. **⭐⭐⭐ Chunk size 2000 (+1.8% MRR)** - Better context preservation
4. **⭐⭐ K=9 (+0.2% MRR)** - Optimal balance
5. **⭐ Enhanced metadata** - Minimal retrieval impact, helps answer formatting

### **Lessons Learned:**

1. **Embeddings matter most:** Single biggest improvement came from GTE-small
2. **Bigger chunks work better:** With excellent embeddings, larger chunks (2000) preserve context
3. **Quality > Quantity:** Both query expansion and hybrid search added more docs but hurt quality
4. **Reranking is valuable:** Even with great embeddings, cross-encoder adds precision
5. **Simple is often better:** Our "simple" semantic search outperformed complex multi-query approaches

### **What Didn't Work:**

- ❌ **Query Expansion** (-1.7% accuracy): More queries = more noise
- ❌ **Hybrid Search** (-2.8% completeness): BM25 diluted semantic quality
- ❌ **Chunk size 1200**: Too small for GTE-small's capabilities
- ❌ **K=20 without reranking**: Too much context overwhelmed LLM

### **Performance Summary:**

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| MRR | 0.7228 | **0.9280** | **+28.4%** |
| Accuracy | 4.00/5.00 | **4.69/5.00** | **+17.3%** |
| Completeness | 3.35/5.00 | **3.98/5.00** | **+18.8%** |
| Relevance | 4.72/5.00 | **4.94/5.00** | **+4.7%** |
| Coverage | 80.8% | **97.1%** | **+20.2%** |

### **Production Readiness:**

✅ **Ready for deployment** with the Test5 configuration
- Excellent retrieval (MRR 0.93)
- High answer quality (Accuracy 4.69/5.00)
- Near-complete coverage (97.1%)
- Fast and stable (no complex multi-stage pipelines)
- Well-documented and reproducible

---

## Future Improvements (If Needed)

If further improvements are required, consider:

1. **Fine-tune GTE-small** on your specific domain
2. **Semantic caching** to reduce LLM calls for similar questions
3. **Question classification** to route different query types appropriately
4. **Metadata filtering** for targeted searches (e.g., "employees in California")
5. **Context compression** using LLM to summarize less relevant chunks

