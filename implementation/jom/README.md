## Experiments Tried

- **Parent chunking:** No significant impact observed—likely because the relevant information was already available from child chunks.
- **Semantic chunking:** Experimented with embedding-based chunking (resulting in 1600 chunks) using a different strategy, but saw no improvement.
- **Different Top_K values:** Only minimal variations in results.
- **Alternative similarity metrics for Chroma (cosine, thresholding):** No measurable benefit.
- **Custom hybrid search:** Implemented a personalized hybrid retrieval method; no improvement detected.
- **Ensemble hybrid search (with varied weights):** Used LangChain's abstraction and found some improvement with different weightings (e.g., 0.25/0.75).
- **Prompt engineering (metadata):** Enriched prompts with metadata and included instructions as well as SQL context.
- **SQL Tool:** Developed a SQL database and instructed the LLM to query it as needed (including always querying), but this did not improve the results.

---

All experiment executions and results are available at: [Excalidraw Experiment Log](https://excalidraw.com/#json=pctKKyMTMs7Ea8yIyxlg1,Poyy2taXlbwnyNOKB3oeAw)