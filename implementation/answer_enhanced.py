from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List
import re
import numpy as np

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# NFL team's proven optimal settings
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
RETRIEVAL_K = 20  # NFL team's optimal: retrieve more candidates
RERANK_K = 9      # NFL team's optimal: final number after reranking

# Enhanced settings for our improvements
USE_QUERY_EXPANSION = True
USE_DYNAMIC_K = True
USE_CONTEXT_COMPRESSION = True

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

# Initialize cross-encoder for reranking (NFL team's approach)
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def advanced_query_expansion(question: str) -> List[str]:
    """
    Enhanced query expansion with domain-specific knowledge and LLM-based expansion.
    """
    # Our original domain-specific expansions
    expansions = {
        "award": ["prize", "recognition", "honor", "achievement", "accolade", "trophy"],
        "employee": ["staff", "worker", "team member", "personnel", "colleague"],
        "founded": ["established", "created", "started", "launched", "began"],
        "contract": ["agreement", "deal", "partnership", "arrangement", "pact"],
        "product": ["service", "offering", "solution", "platform", "tool"],
        "company": ["organization", "firm", "business", "corporation", "enterprise"],
        "year": ["annual", "2023", "2024", "2025", "yearly"],
        "location": ["office", "headquarters", "base", "site", "facility"],
        "salary": ["pay", "wage", "compensation", "income", "earnings"],
        "contract": ["agreement", "deal", "partnership", "arrangement"],
        "insurance": ["coverage", "policy", "protection", "assurance"],
        "technology": ["tech", "software", "system", "platform", "solution"]
    }
    
    expanded_queries = [question]
    question_lower = question.lower()
    
    # Apply domain-specific expansions
    for key, synonyms in expansions.items():
        if key in question_lower:
            for synonym in synonyms:
                expanded_query = question_lower.replace(key, synonym)
                if expanded_query != question_lower and expanded_query not in expanded_queries:
                    expanded_queries.append(expanded_query)
    
    # Add question variations for better coverage
    variations = [
        question_lower.replace("?", "").strip(),
        question_lower.replace("who", "what person").replace("?", "").strip(),
        question_lower.replace("when", "what time").replace("?", "").strip(),
        question_lower.replace("where", "what location").replace("?", "").strip(),
        question_lower.replace("how many", "what number of").replace("?", "").strip(),
    ]
    
    for variation in variations:
        if variation and variation not in expanded_queries:
            expanded_queries.append(variation)
    
    return expanded_queries[:5]  # Limit to 5 queries to avoid too much noise


def calculate_query_complexity(question: str) -> str:
    """
    Determine query complexity to adjust retrieval strategy.
    """
    question_lower = question.lower()
    
    # Simple factual questions
    if any(word in question_lower for word in ["who", "when", "where", "what is", "how many"]):
        if len(question.split()) <= 8:
            return "simple"
    
    # Complex analytical questions
    if any(word in question_lower for word in ["compare", "analyze", "explain", "describe", "why", "how"]):
        return "complex"
    
    # Questions requiring multiple pieces of information
    if any(word in question_lower for word in ["and", "both", "all", "every", "each"]):
        return "multi_fact"
    
    return "medium"


def dynamic_k_selection(question: str, docs: List[Document]) -> int:
    """
    Dynamically adjust the number of documents based on query complexity and quality.
    """
    if not USE_DYNAMIC_K:
        return RERANK_K
    
    complexity = calculate_query_complexity(question)
    
    # Calculate document quality scores
    query_words = set(re.findall(r'\b\w+\b', question.lower()))
    quality_scores = []
    
    for doc in docs:
        content_words = set(re.findall(r'\b\w+\b', doc.page_content.lower()))
        overlap = len(query_words.intersection(content_words))
        quality_scores.append(overlap / len(query_words) if query_words else 0)
    
    avg_quality = np.mean(quality_scores) if quality_scores else 0
    
    # Adjust K based on complexity and quality
    if complexity == "simple" and avg_quality > 0.3:
        return min(5, len(docs))  # Fewer docs for simple, high-quality matches
    elif complexity == "complex":
        return min(12, len(docs))  # More docs for complex questions
    elif complexity == "multi_fact":
        return min(10, len(docs))  # Moderate docs for multi-fact questions
    else:
        return min(RERANK_K, len(docs))  # Default behavior


def compress_context(docs: List[Document], question: str) -> List[Document]:
    """
    Compress context by removing redundant information and focusing on relevant parts.
    """
    if not USE_CONTEXT_COMPRESSION or len(docs) <= 3:
        return docs
    
    query_words = set(re.findall(r'\b\w+\b', question.lower()))
    compressed_docs = []
    
    for doc in docs:
        # Extract the most relevant sentences
        sentences = doc.page_content.split('. ')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append(sentence)
        
        # If we found relevant sentences, create a compressed version
        if relevant_sentences:
            compressed_content = '. '.join(relevant_sentences[:3])  # Limit to top 3 sentences
            if len(compressed_content) < len(doc.page_content) * 0.7:  # Only if significantly shorter
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata=doc.metadata
                )
                compressed_docs.append(compressed_doc)
            else:
                compressed_docs.append(doc)
        else:
            compressed_docs.append(doc)
    
    return compressed_docs


def rerank_documents_cross_encoder(query: str, documents: List[Document]) -> List[Document]:
    """
    NFL team's cross-encoder reranking with our enhancements.
    """
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score and return top documents
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Apply dynamic K selection
    dynamic_k = dynamic_k_selection(query, documents)
    top_docs = [doc for doc, score in doc_scores[:dynamic_k]]
    
    # Apply context compression
    compressed_docs = compress_context(top_docs, query)
    
    return compressed_docs


def format_doc_with_metadata(doc: Document, idx: int) -> str:
    """
    NFL team's metadata formatting with our enhancements.
    """
    meta = doc.metadata
    formatted = f"--- Document {idx+1} ---\n"
    
    # Add structured metadata first
    if 'entity_name' in meta:
        formatted += f"Entity: {meta['entity_name']}\n"
    if 'doc_type' in meta:
        formatted += f"Type: {meta['doc_type']}\n"
    if 'job_title' in meta:
        formatted += f"Job Title: {meta['job_title']}\n"
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
    
    # Add content with relevance highlighting
    content = doc.page_content
    query_words = set(re.findall(r'\b\w+\b', meta.get('title', '').lower()))
    if query_words:
        # Highlight relevant terms in content
        for word in query_words:
            if len(word) > 3:  # Only highlight longer words
                content = re.sub(f'\\b{word}\\b', f'**{word}**', content, flags=re.IGNORECASE)
    
    formatted += f"\nContent:\n{content}\n"
    return formatted


def fetch_context(question: str) -> list[Document]:
    """
    Enhanced retrieval: NFL team's approach + our query expansion + dynamic optimization.
    """
    if USE_QUERY_EXPANSION:
        # Use our enhanced query expansion
        expanded_queries = advanced_query_expansion(question)
        
        # Retrieve documents for each expanded query
        all_documents = []
        for query in expanded_queries:
            docs = retriever.invoke(query, k=RETRIEVAL_K)
            all_documents.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_documents:
            doc_id = (doc.page_content, doc.metadata.get('source', ''))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
    else:
        # Use single query retrieval
        unique_docs = retriever.invoke(question, k=RETRIEVAL_K)
    
    # Use NFL team's cross-encoder reranking with our enhancements
    return rerank_documents_cross_encoder(question, unique_docs)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Enhanced answer generation with NFL team's approach + our improvements.
    """
    docs = fetch_context(question)
    
    # Format with enhanced metadata for LLM
    context = "\n\n".join(format_doc_with_metadata(doc, i) for i, doc in enumerate(docs))
    
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
