#!/usr/bin/env python3
"""
Hyperparameter Optimization for RAG
Optimizes: chunk_size, K value, embedding models
Uses: 30 sampled questions for speed
"""

import os
import sys
import time
import glob
import re
import random
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Non-interactive backend
matplotlib.use('Agg') 

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.test import load_tests
from evaluation.eval import calculate_mrr, calculate_ndcg

# Configuration
SAMPLE_SIZE = 30 
KNOWLEDGE_BASE = str(Path(__file__).parent.parent.parent / "knowledge-base")

# ==============================================================================
# 1. Sample Test Questions
# ==============================================================================
all_tests = list(load_tests())
print(f"Total tests: {len(all_tests)}")

# Stratified sampling
random.seed(42)
by_category = {}
for test in all_tests:
    if test.category not in by_category:
        by_category[test.category] = []
    by_category[test.category].append(test)

sampled_tests = []
for category, tests in by_category.items():
    n = max(2, int(len(tests) / len(all_tests) * SAMPLE_SIZE))
    sampled_tests.extend(random.sample(tests, min(n, len(tests))))

sampled_tests = sampled_tests[:SAMPLE_SIZE]

print(f"Sampled {len(sampled_tests)} questions by category:")
for cat, tests in by_category.items():
    count = sum(1 for t in sampled_tests if t.category == cat)
    if count > 0:
        print(f"  {cat}: {count}")

# ==============================================================================
# 2. Load Documents
# ==============================================================================
print("\n2. LOADING DOCUMENTS")
print("-" * 80)

def extract_metadata(doc, folder):
    """Extract rich metadata"""
    metadata = doc.metadata.copy()
    doc_type = os.path.basename(folder)
    filename = Path(doc.metadata['source']).stem
    
    metadata['entity_name'] = filename
    metadata['doc_type'] = doc_type
    lines = doc.page_content.split('\n')
    metadata['title'] = lines[0].replace('#', '').strip() if lines else ''
    
    if doc_type == 'employees':
        salary_match = re.search(r'\$[\d,]+', doc.page_content)
        if salary_match:
            metadata['salary'] = salary_match.group()
    
    return metadata

def load_documents():
    folders = glob.glob(f"{KNOWLEDGE_BASE}/*")
    documents = []
    for folder in folders:
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata = extract_metadata(doc, folder)
            documents.append(doc)
    return documents

documents = load_documents()
print(f"✓ Loaded {len(documents)} documents")

# ==============================================================================
# 3. Experiment 1: Chunk Size Optimization
# ==============================================================================
print("\n3. EXPERIMENT 1: CHUNK SIZE OPTIMIZATION")
print("-" * 80)

def create_chunks_with_size(documents, chunk_size, chunk_overlap_ratio=0.2):
    overlap = int(chunk_size * chunk_overlap_ratio)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def evaluate_chunks(chunks, embeddings, tests, k=5):
    db_path = f"temp_db_{int(time.time())}"
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=db_path
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    mrr_scores = []
    ndcg_scores = []
    
    for test in tests:
        docs = retriever.invoke(test.question)
        mrr = np.mean([calculate_mrr(kw, docs) for kw in test.keywords])
        ndcg = np.mean([calculate_ndcg(kw, docs, k) for kw in test.keywords])
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    return np.mean(mrr_scores), np.mean(ndcg_scores)

# Test chunk sizes with GTE-small
chunk_sizes = [800, 1000, 1200, 1500, 2000]
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
print("Using GTE-small embeddings for testing...")

chunk_results = []
for size in chunk_sizes:
    print(f"\nChunk size: {size}")
    start = time.time()
    
    chunks = create_chunks_with_size(documents, size)
    print(f"  Created {len(chunks)} chunks")
    
    mrr, ndcg = evaluate_chunks(chunks, embeddings, sampled_tests, k=5)
    elapsed = time.time() - start
    
    chunk_results.append({
        'chunk_size': size,
        'num_chunks': len(chunks),
        'mrr': mrr,
        'ndcg': ndcg,
        'time_seconds': elapsed
    })
    print(f"  MRR: {mrr:.4f}, nDCG: {ndcg:.4f}, Time: {elapsed:.1f}s")

chunk_df = pd.DataFrame(chunk_results)
best_chunk = chunk_df.loc[chunk_df['mrr'].idxmax()]
print(f"\n🏆 Best chunk size: {best_chunk['chunk_size']} (MRR: {best_chunk['mrr']:.4f})")

# Plot chunk size results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Chunk Size Optimization Results', fontsize=16, fontweight='bold')

# MRR
axes[0, 0].plot(chunk_df['chunk_size'], chunk_df['mrr'], 'o-', linewidth=2, markersize=8, color='blue')
axes[0, 0].set_xlabel('Chunk Size (chars)')
axes[0, 0].set_ylabel('MRR')
axes[0, 0].set_title('MRR by Chunk Size')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(best_chunk['chunk_size'], color='red', linestyle='--', alpha=0.7, label='Best')
axes[0, 0].legend()

# nDCG
axes[0, 1].plot(chunk_df['chunk_size'], chunk_df['ndcg'], 'o-', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Chunk Size (chars)')
axes[0, 1].set_ylabel('nDCG')
axes[0, 1].set_title('nDCG by Chunk Size')
axes[0, 1].grid(True, alpha=0.3)

# Number of chunks
axes[1, 0].bar(chunk_df['chunk_size'], chunk_df['num_chunks'], color='orange', alpha=0.7)
axes[1, 0].set_xlabel('Chunk Size (chars)')
axes[1, 0].set_ylabel('Number of Chunks')
axes[1, 0].set_title('Chunks Created')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Processing time
axes[1, 1].bar(chunk_df['chunk_size'], chunk_df['time_seconds'], color='red', alpha=0.7)
axes[1, 1].set_xlabel('Chunk Size (chars)')
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].set_title('Processing Time')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('implementation/dkisselev-zz/hyperparameter_chunk_size.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot: implementation/dkisselev-zz/hyperparameter_chunk_size.png")
plt.close()

# ==============================================================================
# 4. Experiment 2: K Value Optimization
# ==============================================================================

best_chunk_size = int(best_chunk['chunk_size'])
best_chunks = create_chunks_with_size(documents, best_chunk_size)

print(f"Creating vector store with {len(best_chunks)} chunks...")
db_path = "temp_k_optimization"
vectorstore = Chroma.from_documents(
    documents=best_chunks,
    embedding=embeddings,
    persist_directory=db_path
)

k_values = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
k_results = []

for k in k_values:
    print(f"\nK = {k}")
    start = time.time()
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    mrr_scores = []
    ndcg_scores = []
    coverage_scores = []
    
    for test in sampled_tests:
        docs = retriever.invoke(test.question)
        mrr = np.mean([calculate_mrr(kw, docs) for kw in test.keywords])
        ndcg = np.mean([calculate_ndcg(kw, docs, k) for kw in test.keywords])
        
        all_text = " ".join([d.page_content.lower() for d in docs])
        found = sum(1 for kw in test.keywords if kw.lower() in all_text)
        coverage = (found / len(test.keywords)) * 100
        
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        coverage_scores.append(coverage)
    
    elapsed = time.time() - start
    
    k_results.append({
        'k': k,
        'mrr': np.mean(mrr_scores),
        'ndcg': np.mean(ndcg_scores),
        'coverage': np.mean(coverage_scores),
        'time_seconds': elapsed
    })
    print(f"  MRR: {np.mean(mrr_scores):.4f}, Coverage: {np.mean(coverage_scores):.1f}%")

if os.path.exists(db_path):
    shutil.rmtree(db_path)

k_df = pd.DataFrame(k_results)
best_k = k_df.loc[k_df['mrr'].idxmax()]
print(f"\n🏆 Best K value: {best_k['k']} (MRR: {best_k['mrr']:.4f})")

# Plot K value results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('K Value Optimization Results', fontsize=16, fontweight='bold')

# MRR
axes[0].plot(k_df['k'], k_df['mrr'], 'o-', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('K Value')
axes[0].set_ylabel('MRR')
axes[0].set_title('MRR by K')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(best_k['k'], color='red', linestyle='--', alpha=0.7, label='Best')
axes[0].legend()

# nDCG
axes[1].plot(k_df['k'], k_df['ndcg'], 'o-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('K Value')
axes[1].set_ylabel('nDCG')
axes[1].set_title('nDCG by K')
axes[1].grid(True, alpha=0.3)

# Coverage
axes[2].plot(k_df['k'], k_df['coverage'], 'o-', linewidth=2, markersize=8, color='orange')
axes[2].set_xlabel('K Value')
axes[2].set_ylabel('Coverage (%)')
axes[2].set_title('Keyword Coverage by K')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('implementation/dkisselev-zz/hyperparameter_k_value.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot: implementation/dkisselev-zz/hyperparameter_k_value.png")
plt.close()

# ==============================================================================
# 5. Experiment 3: Embedding Model Comparison (CPU-friendly)
# ==============================================================================

embedding_models = [
    ("all-MiniLM-L6-v2", "MiniLM-L6 (current, 384d)"),
    ("all-MiniLM-L12-v2", "MiniLM-L12 (384d)"),
    ("all-mpnet-base-v2", "MPNet-base (768d)"),
    ("thenlper/gte-small", "GTE-small (384d)"),
]

best_k_value = int(best_k['k'])
embedding_results = []

print(f"Testing with chunk_size={best_chunk_size}, k={best_k_value}")

for model_name, description in embedding_models:
    print(f"\n{description}")
    start = time.time()
    
    try:
        emb = HuggingFaceEmbeddings(model_name=model_name)
        
        db_path = f"temp_emb_{int(time.time())}"
        vectorstore = Chroma.from_documents(
            documents=best_chunks,
            embedding=emb,
            persist_directory=db_path
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": best_k_value})
        
        mrr_scores = []
        ndcg_scores = []
        
        for test in sampled_tests:
            docs = retriever.invoke(test.question)
            mrr = np.mean([calculate_mrr(kw, docs) for kw in test.keywords])
            ndcg = np.mean([calculate_ndcg(kw, docs, best_k_value) for kw in test.keywords])
            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
        
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        elapsed = time.time() - start
        
        embedding_results.append({
            'model': description,
            'model_name': model_name,
            'mrr': np.mean(mrr_scores),
            'ndcg': np.mean(ndcg_scores),
            'time_seconds': elapsed
        })
        print(f"  MRR: {np.mean(mrr_scores):.4f}, Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

embedding_df = pd.DataFrame(embedding_results)
if len(embedding_df) > 0:
    best_emb = embedding_df.loc[embedding_df['mrr'].idxmax()]
    print(f"\n🏆 Best embedding: {best_emb['model']} (MRR: {best_emb['mrr']:.4f})")
    
    # Plot embedding results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Embedding Model Comparison', fontsize=16, fontweight='bold')
    
    # MRR
    bars1 = axes[0].bar(range(len(embedding_df)), embedding_df['mrr'], color='blue', alpha=0.7)
    axes[0].set_xticks(range(len(embedding_df)))
    axes[0].set_xticklabels(embedding_df['model'], rotation=45, ha='right')
    axes[0].set_ylabel('MRR')
    axes[0].set_title('MRR by Embedding Model')
    axes[0].grid(True, alpha=0.3, axis='y')
    # Highlight best
    best_idx = embedding_df['mrr'].idxmax()
    bars1[best_idx].set_color('green')
    
    # Processing time
    bars2 = axes[1].bar(range(len(embedding_df)), embedding_df['time_seconds'], color='red', alpha=0.7)
    axes[1].set_xticks(range(len(embedding_df)))
    axes[1].set_xticklabels(embedding_df['model'], rotation=45, ha='right')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Processing Time')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('implementation/dkisselev-zz/hyperparameter_embeddings.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot: implementation/dkisselev-zz/hyperparameter_embeddings.png")
    plt.close()
else:
    best_emb = None
    print("\n⚠️  No embeddings tested successfully")

# ==============================================================================
# 6. Final Summary
# ==============================================================================

print(f"\n🔹 Best Chunk Size: {best_chunk['chunk_size']}")
print(f"   MRR: {best_chunk['mrr']:.4f}")
print(f"   Chunks: {best_chunk['num_chunks']}")

print(f"\n🔹 Best K Value: {best_k['k']}")
print(f"   MRR: {best_k['mrr']:.4f}")
print(f"   Coverage: {best_k['coverage']:.1f}%")

if best_emb is not None:
    print(f"\n🔹 Best Embedding: {best_emb['model']}")
    print(f"   MRR: {best_emb['mrr']:.4f}")
    print(f"   Time: {best_emb['time_seconds']:.1f}s")

# Save results
results = {
    'best_chunk_size': int(best_chunk['chunk_size']),
    'best_k': int(best_k['k']),
    'chunk_results': chunk_results,
    'k_results': k_results,
    'embedding_results': embedding_results if best_emb is not None else []
}

if best_emb is not None:
    results['best_embedding'] = best_emb['model_name']
    results['final_mrr'] = float(best_emb['mrr'])

output_file = 'implementation/dkisselev-zz/hyperparameter_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)


# Print tables
print("\nChunk Size Results:")
print(chunk_df.to_string(index=False))

print("\nK Value Results:")
print(k_df.to_string(index=False))

if len(embedding_df) > 0:
    print("\nEmbedding Results:")
    print(embedding_df.to_string(index=False))
