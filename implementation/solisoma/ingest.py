from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import re

from dotenv import load_dotenv


load_dotenv(override=True)


MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
FILLER_WORDS = [
    "what is", "what are", "how to", "how do i", "tell me", "who", "when did", "how many"
    "can you", "could you", "please", "explain", "describe",
    "i want to know", "show me", "give me", "define", "in detail",
    "right now", "basically", "just", "currently"
]

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""
RETRIEVAL_K = 30
RERANK_TOP_K = 12 

# embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def clean_q(q: str) -> str:
    """
    Clean the question by removing filler words and other noise.
    """
    q = q.lower()
    for phrase in FILLER_WORDS:
        q = re.sub(rf"\b{re.escape(phrase)}\b", "", q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def format_context(docs: list[Document]) -> str:
    """
    Combine retrieved documents into a formatted context string
    with helpful metadata for traceability.
    """
    formatted = []
    for doc in docs:
        meta = doc.metadata
        source = Path(meta.get("filename", "Unknown")).name
        doc_type = meta.get("doc_type", "general")
        chunk = meta.get("chunk_index", "?")

        header = f"[Source: {source} | Type: {doc_type} | Chunk: {chunk}]"
        formatted.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted)


def fetch_context(q: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    query = clean_q(q)
    
    docs = retriever.invoke(q, k=RETRIEVAL_K)
    
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:RERANK_TOP_K]]

    return top_docs


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    context = format_context(docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question)) 
    response = llm.invoke(messages)
    return response.content, docs


answer_question("what is the purpose of insurellm?")
