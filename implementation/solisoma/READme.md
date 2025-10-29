# Solisoma RAG Implementation

## Overview

This implementation enhances the RAG (Retrieval-Augmented Generation) system with improved embeddings, query preprocessing, cross-encoder reranking, and optimized chunking strategies to improve retrieval accuracy (MRR) and answer quality.

## Key Improvements

### 1. Enhanced Embedding Model
- **Model**: `thenlper/gte-small`
- **Why**: Better semantic understanding compared to `all-MiniLM-L6-v2`
- **Impact**: Improved retrieval quality through more accurate vector representations

### 2. Query Preprocessing (`clean_q()`)
- **Feature**: Removes filler words and noise from queries
- **Purpose**: Focus on core query intent for better retrieval
- **Examples**: Removes "what is", "how to", "tell me", "please", etc.
- **Impact**: More focused semantic search on the vectorstore, especially for conversational queries

### 3. Cross-Encoder Reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Process**: 
  - Retrieves 30 candidates via semantic search
  - Reranks using cross-encoder (question-document pairs)
  - Returns top 12 documents
- **Why**: Cross-encoders understand question-document relationships better than simple similarity
- **Impact**: Significantly improved relevance ranking (better MRR scores)

### 4. Optimized Chunking Strategy (`ingest.py`)
- **Chunk Size**: 2048 characters (vs default 1000) - chosen as a power of 2 (2¹¹ = 2048), similar to memory sizes (1KB=1024, 2KB=2048), which is computationally efficient
- **Overlap**: 400 characters (~20% overlap)
- **Separators**: Hierarchical splitting strategy
  ```
  ["\n## ", "\n### ", "\n#### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]
  ```
- **Why**: 
  - Larger chunks preserve more context
  - Hierarchical separators respect document structure (headers first)
  - Better overlap ensures no information loss at boundaries
- **Metadata**: Each chunk tracks `chunk_index` for traceability

### 5. Enhanced Metadata Tracking
- **Tracks**:
  - `doc_type`: Source folder (employees, contracts, products, company)
  - `source`: Full file path
  - `filename`: Just the filename
  - `chunk_index`: Position of chunk in document
- **Purpose**: Better context formatting and source attribution

### 6. Structured Context Formatting (`format_context()`)
- **Format**: Includes source, type, and chunk information
- **Structure**: 
  ```
  [Source: filename.md | Type: doc_type | Chunk: index]
  [document content]
  ```
- **Why**: Helps LLM understand context provenance and improves answer accuracy

## Configuration

### Key Parameters (`answer.py`)

| Parameter      | Value          | Purpose                                   |
|----------------|----------------|-------------------------------------------|
| `RETRIEVAL_K`  | 30             | Number of initial candidates retrieved    |
| `RERANK_TOP_K` | 12             | Final number of documents after reranking |
| `MODEL`        | "gpt-4.1-nano" | LLM for answer generation                 |

### Chunking Parameters (`ingest.py`)

| Parameter       | Value | Purpose                               |
|-----------------|-------|---------------------------------------|
| `chunk_size`    | 2048  | Maximum characters per chunk          |
| `chunk_overlap` | 400   | Characters overlapping between chunks |

## How It Works

### Retrieval Pipeline

1. **Query Preprocessing**
   ```
   Original: "What is the purpose of Insurellm?"
   Cleaned: "purpose of Insurellm"
   ```

2. **Semantic Retrieval**
   - Uses cleaned query to retrieve top 30 candidates
   - `gte-small` embeddings find semantically similar documents

3. **Cross-Encoder Reranking**
   - Creates question-document pairs
   - Scores each pair for relevance
   - Ranks by relevance score
   - Returns top 12 documents

4. **Context Formatting**
   - Structures documents with metadata
   - Includes source, type, and chunk information
   - Formats for LLM consumption

5. **Answer Generation**
   - LLM receives structured context
   - Generates answer based on retrieved documents

## Performance Optimizations

1. **Two-Stage Retrieval**: Broad candidate pool (30) → Precise reranking (12)
2. **Query Cleaning**: Reduces noise in semantic search
3. **Hierarchical Chunking**: Preserves document structure
4. **Structured Context**: Improves LLM answer quality

