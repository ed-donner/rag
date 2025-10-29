import glob
import os
from pathlib import Path
import pickle
import re
import time
from typing import Dict, List
import uuid
import uuid

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



start_time = time.time()



MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
BM25_DB_NAME = str(Path(__file__).parent.parent / "bm25_index.pkl")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
DB_PATH = str(Path(__file__).parent.parent / "sqldb.db")


#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")     # Higher quality

load_dotenv(override=True)




def generate_doc_id(content: str, metadata: dict) -> str:
    """Generate unique ID for document"""
    unique_str = f"{content[:100]}{metadata.get('source', '')}"
    # UUID5 uses SHA-1 hashing and is deterministic
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

def fetch_documents_hierarchical():
    """
    Fetch documents with hierarchical structure:
    - Parent docs: Aggregated folder content
    - Child docs: Individual file chunks
    """
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    all_documents = []
    parent_documents = []
    
    for folder in folders:
        doc_type = os.path.basename(folder)
        
        # Load all documents in folder
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        folder_docs = loader.load()
                
        # Create parent document (aggregated folder content)
        combined_content = "\n\n".join([doc.page_content for doc in folder_docs])
        parent_id = f"parent_{doc_type}"
        
        parent_doc = Document(
            page_content=combined_content[:10000],  # Limit parent size
            metadata={
                "doc_type": doc_type,
                "is_parent": True,
                "parent_id": parent_id,
                "source": folder,
                "child_count": len(folder_docs)
            }
        )
        parent_documents.append(parent_doc)
        
        # Create child documents with parent reference
        for doc in folder_docs:
            doc_id = generate_doc_id(doc.page_content, doc.metadata)
            doc.metadata.update({
                "doc_type": doc_type,
                "is_parent": False,
                "parent_id": parent_id,
                "doc_id": doc_id
            })
            all_documents.append(doc)
    
    return all_documents, parent_documents

def create_hierarchical_chunks(documents: List[Document], parent_documents: List[Document]):
    """
    Create chunks with hierarchical relationships:
    - Parent chunks: Larger summaries (2000 tokens)
    - Child chunks: Detailed content (1000 tokens)
    """
    # Chunk parent documents (larger chunks for context)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # Prefer splitting on headers
        is_separator_regex=False

    )
    parent_chunks = parent_splitter.split_documents(parent_documents)
    
    # Chunk child documents (standard chunks)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # Prefer splitting on headers
        is_separator_regex=False

    )
    child_chunks = child_splitter.split_documents(documents)
    
    # Add chunk indices for tracking
    for i, chunk in enumerate(parent_chunks):
        chunk.metadata["chunk_index"] = i
        
    for i, chunk in enumerate(child_chunks):
        chunk.metadata["chunk_index"] = i
    
    return parent_chunks, child_chunks

def create_semantic_chunks(documents, model_name='mixedbread-ai/mxbai-embed-large-v1', similarity_threshold=0.6, max_sentences=8):
    """
    Semantic chunking: embed first, then split based on semantic coherence.
    Retains RecursiveCharacterTextSplitter as a fallback for structure-aware splitting.
    """

    model = SentenceTransformer(model_name)
    semantic_chunks = []

    for doc in documents:
        text = doc.page_content.strip()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            continue

        embeddings = model.encode(sentences)
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]

            if sim < similarity_threshold or len(current_chunk) >= max_sentences:
                chunk_text = '. '.join(current_chunk) + '.'
                semantic_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            semantic_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))

    # Optional: apply a structural splitter as a final refinement
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        is_separator_regex=False
    )

    final_chunks = text_splitter.split_documents(semantic_chunks)
    return final_chunks

def create_bm25_index(chunks: List[Document]):
    """Create BM25 sparse index for keyword search with minimal preprocessing"""
    
    # Small helper function - just cleans and tokenizes better
    def tokenize(text):
        # Remove special characters, keep alphanumeric
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        # Split and filter very short tokens (noise)
        return [t for t in text.split() if len(t) > 2]
    
    # Use improved tokenization instead of just .lower().split()
    tokenized_corpus = [tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index and document mapping
    with open(BM25_DB_NAME, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'chunks': chunks,
            'tokenized_corpus': tokenized_corpus
        }, f)
    
    print(f"BM25 index created with {len(chunks)} documents")
    return bm25

def create_vector_store(parent_chunks: List[Document], child_chunks: List[Document]):
    """Create dense vector store with both parent and child chunks"""
    # Combine all chunks
    all_chunks = parent_chunks + child_chunks
    
    # Clear existing database
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    
    print(f"Vector store created:")
    print(f"  - Total chunks: {count:,}")
    print(f"  - Parent chunks: {len(parent_chunks):,}")
    print(f"  - Child chunks: {len(child_chunks):,}")
    print(f"  - Embedding dimensions: {dimensions:,}")
    
    return vectorstore

if __name__ == "__main__":
    print("Starting hierarchical hybrid ingestion...")
    
    documents, parent_documents = fetch_documents_hierarchical()
    print(f"Loaded {len(documents)} child documents and {len(parent_documents)} parent documents")
    
    parent_chunks, child_chunks = create_hierarchical_chunks(documents, parent_documents)
    #child_chunks = create_semantic_chunks(documents)
    print(f"Created {len([])} parent chunks and {len(child_chunks)} child chunks")
    
    vectorstore = create_vector_store([], child_chunks) #create_vector_store(parent_chunks, child_chunks)
    
    all_chunks = [] + child_chunks #parent_chunks + child_chunks
    create_bm25_index(all_chunks)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Ingestion complete! Total time: {elapsed:.2f} seconds")

