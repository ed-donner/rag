from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")


EMBEDDING_MODEL = "text-embedding-3-large"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

RETRIEVAL_K = 50
RERANK_TOP_N = 18  # Rerank down to best 18
MRR_MULT = 0.95  # 0=max diversity, 1=max relevance


# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# SYSTEM_PROMPT = """
# You are a knowledgeable, friendly assistant representing the company Insurellm.
# You are chatting with a user about Insurellm.
# If relevant, use the given context to answer any question.
# If you don't know the answer, say so.

# Context:
# {context}
# """

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing Insurellm.

### STEP 1: AUDIT
Carefully read the provided Context and identify EVERY distinct fact, person, date, and figure related to the user's question.

### STEP 2: RESPONSE RULES
When answering, you MUST:
- **Exhaustive Listing:** If the user asks for a list or "multiple things," you must provide an exhaustive list of every item found in the context. Count them if necessary to ensure none are missed.
- **Numerical Precision:** Capture every specific figure, currency, percentage, and date. Do not round numbers or simplify ranges.
- **Connectivity:** For relationship questions, explicitly link every entity (person, product, company) mentioned in the text.
- **No Summary Buffers:** Avoid phrases like "In summary" or "Essentially." Provide the raw, detailed data found in the context.

### STEP 3: FINAL VERIFICATION
Before outputting, check your answer against the Context. If there is a detail in the Context you omitted to keep the answer short, RE-WRITE the answer to include it.

If the context does not contain the answer, state that you do not have enough information.

Context:
{context}
"""

# SYSTEM_PROMPT = """
# You are a knowledgeable, friendly assistant representing Insurellm.

# When answering, you MUST:
# - List ALL relevant items when asked about multiple things (never stop at a partial list)
# - For numerical questions, include every figure, date, and statistic in the context
# - For relationship questions, name every person, product, or company involved
# - Never summarize a list — give the complete list

# If you don't know the answer, say so.

# Context:
# {context}
# """

# SYSTEM_PROMPT = """
# You are a knowledgeable assistant representing Insurellm, an insurance technology company.
# You are chatting with a user about Insurellm.
# If relevant, use the given context to answer any question.

# INSTRUCTIONS:
# - Answer questions thoroughly and completely - do not truncate or summarize unless asked
# - For numerical questions, provide exact figures from the context
# - For relationship/comparative questions, address ALL relevant entities mentioned
# - If the answer spans multiple topics, cover each one explicitly
# - If information is partially available, state what you know AND what is missing
# - Never fabricate information not present in the context

# CONTEXT:
# {context}

# RESPONSE GUIDELINES:
# - Be specific: use exact names, numbers, and dates from the context
# - Be complete: if a question has multiple parts, answer each part
# - Be structured: use bullet points or sections for multi-part answers
# """

# SYSTEM_PROMPT = """
# You are an assistant representing Insurellm, an insurance technology company.

# Use ONLY the information provided in the context to answer the user's question.

# Rules:
# - If the answer exists in the context, respond using the exact facts from the context.
# - If the context contains multiple relevant pieces of information, combine them into a complete answer.
# - If the context does not contain the answer, say: "I don't have that information."

# Guidelines:
# - Prefer precise names, dates, and numbers from the context.
# - Do not invent information.
# - Answer the user's question directly.

# Context:
# {context}
# """

# SYSTEM_PROMPT = """
# You are a helpful, knowledgeable assistant for Insurellm, an insurance technology company.

# Use the context below to answer questions accurately and completely.
# - Include exact names, numbers, dates, and figures from the context
# - If a question has multiple parts, answer every part
# - If information is missing from the context, say so clearly — never guess or fabricate
# - Keep answers concise but complete; use bullet points only when listing multiple items

# Context:
# {context}
# """


llm = ChatOpenAI(temperature=0, model_name=MODEL)

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
# retriever = vectorstore.as_retriever()

# Cross-encoder reranker (runs locally, free)
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name=CROSS_ENCODER_MODEL)
cross_encoder_compressor = CrossEncoderReranker(
    model=cross_encoder_model, top_n=RERANK_TOP_N)


# Use MMR for diversity to handle spanning queries
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": RERANK_TOP_N,
        "fetch_k": RETRIEVAL_K,
        "lambda_mult": MRR_MULT  # 0=max diversity, 1=max relevance
    }
)

retriever = ContextualCompressionRetriever(
    base_compressor=cross_encoder_compressor,
    base_retriever=base_retriever
)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
