from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
# Path goes: answer.py -> dkisselev-zz -> implementation -> project_root
DB_NAME = str(Path(__file__).parent.parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
RETRIEVAL_K = 20  # Retrieve candidates for reranking
FINAL_K = 9  # Optimal: K=9 per refined hyperparameter tuning
USE_QUERY_EXPANSION = False  # Step 3: Disabled (hurts accuracy -1.7%)
USE_HYBRID_SEARCH = False  # Step 4: Disabled (hurts completeness -2.8%)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context (with metadata):
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatOpenAI(temperature=0, model_name=MODEL)

# Initialize BM25 index for hybrid search
_bm25 = None
_bm25_docs = None

def get_bm25():
    """Initialize BM25 index from all documents in vector store"""
    global _bm25, _bm25_docs
    if _bm25 is None:
        # Get all documents from vector store
        collection = vectorstore._collection
        all_data = collection.get(include=["documents", "metadatas"])
        
        # Create Document objects
        _bm25_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_data['documents'], all_data['metadatas'])
        ]
        
        # Tokenize documents
        tokenized_docs = [doc.page_content.lower().split() for doc in _bm25_docs]
        _bm25 = BM25Okapi(tokenized_docs)
    
    return _bm25, _bm25_docs

# Initialize cross-encoder for reranking
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def expand_query(question: str) -> list[str]:
    """
    Query Expansion: Generate 2-3 variations of the query to improve retrieval coverage.    
    """
    expansion_prompt = f"""Given this question, generate 2 alternative phrasings that would help find relevant information.
Keep the variations concise and focused on the same topic.

Original question: {question}

Provide ONLY 2 alternative phrasings, one per line, without numbering or extra text:"""
    
    try:
        response = llm.invoke([HumanMessage(content=expansion_prompt)])
        variations = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
        # Return original + variations (limit to 3 total)
        return [question] + variations[:2]
    except Exception as e:
        # Fallback: just use original question if expansion fails
        print(f"Query expansion failed: {e}")
        return [question]


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve and rerank relevant context documents for a question.
    GTE-small + (BM25 hybrid search) + reranking + K=9
    """
    # Semantic search (vector similarity)
    semantic_docs = retriever.invoke(question)
    
    # BM25 keyword search
    if USE_HYBRID_SEARCH:
        bm25, bm25_docs = get_bm25()
        
        # Tokenize query
        tokenized_query = question.lower().split()
        
        # Get BM25 scores for all documents
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top K BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_K]
        bm25_results = [bm25_docs[i] for i in top_bm25_indices]
        
        # Combine semantic and BM25 results (deduplicate)
        all_docs = []
        seen_ids = set()
        
        for doc in semantic_docs + bm25_results:
            doc_id = f"{doc.metadata.get('source', '')}:{hash(doc.page_content)}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    else:
        all_docs = semantic_docs
    
    # Rerank all collected documents with cross-encoder
    reranker = get_reranker()
    pairs = [[question, doc.page_content] for doc in all_docs]
    scores = reranker.predict(pairs)
    
    # Sort by score and return top FINAL_K
    doc_scores = list(zip(all_docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in doc_scores[:FINAL_K]]
    
    return top_docs


def format_doc_with_metadata(doc: Document, idx: int) -> str:
    """Format document with metadata"""
    meta = doc.metadata
    formatted = f"--- Document {idx+1} ---\n"
    
    # Add structured metadata first
    if 'entity_name' in meta:
        formatted += f"Entity: {meta['entity_name']}\n"
    if 'doc_type' in meta:
        formatted += f"Type: {meta['doc_type']}\n"
    if 'job_title' in meta:
        formatted += f"Job Title: {meta['job_title']}\n"
    # if 'dob' in meta:
    #     formatted += f"Date of Birth: {meta['dob']}\n"
    if 'salary' in meta:
        formatted += f"Salary: {meta['salary']}\n"
    if 'location' in meta:
        formatted += f"Location: {meta['location']}\n"
    if 'product_name' in meta:
        formatted += f"Product: {meta['product_name']}\n"
    if 'client_name' in meta:
        formatted += f"Client Name: {meta['client_name']}\n"
    if 'contract_number' in meta:
        formatted += f"Contract #: {meta['contract_number']}\n"
    if 'monthly_payment' in meta:
        formatted += f"Payment: {meta['monthly_payment']}\n"
    
    # Add content
    formatted += f"\nContent:\n{doc.page_content}\n"
    return formatted


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    
    # Format with metadata for LLM
    context = "\n\n".join(format_doc_with_metadata(doc, i) for i, doc in enumerate(docs))
    
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
