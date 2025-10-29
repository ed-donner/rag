import json
import os
from pathlib import Path
import pickle
import re
import sqlite3
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv(override=True)


def get_database_schema(db_path: str) -> str:
    """
    Extract database schema information.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        schema_info.append(f"\n### Table: {table_name}")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        schema_info.append("Columns:")
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            not_null = "NOT NULL" if col[3] else ""
            pk = "PRIMARY KEY" if col[5] else ""
            schema_info.append(f"  - {col_name} ({col_type}) {not_null} {pk}".strip())
    
    conn.close()
    return "\n".join(schema_info)

DATABASE_SCHEMA = get_database_schema("sqldb.db")  # Load once at startup


MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
BM25_DB_NAME = str(Path(__file__).parent.parent / "bm25_index.pkl")
DB_PATH = str(Path(__file__).parent.parent / "sqldb.db")

# Use the same embedding model as ingestion
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Retrieval parameters
RETRIEVAL_K = 9  # Increased for better context
# Enhanced system prompt with structured instructions
SYSTEM_PROMPT = """You are an expert AI assistant for Insurellm, a company specializing in insurance technology and solutions.

Your role is to provide accurate, helpful, and professional responses based on the company's knowledge base and a SQL database.

## Instructions:
0. **Extract information from SQL**: Always try to extract information from the SQL database.
1. **Answer from Context**: Prioritize information from the provided context below, combining extracts and SQL information to critically extract the best answer.
2. **Cite Sources**: When using specific information, mention the document type (e.g., "According to our contract documentation...")
3. **Be Honest**: If the context doesn't contain the answer, clearly state "I don't have that information in our knowledge base"
4. **Be Concise**: Provide clear, well-structured answers without unnecessary verbosity.

## SQL Database Schema:
{schema}
## Available Context:
{context}
## Context Metadata:
{metadata}

Remember: Only answer based on the context provided using the most appropriate data source. If you're unsure or the information isn't in the context, say so clearly.
"""

# Alternative prompt for when no context is found
NO_CONTEXT_PROMPT = """You are an expert AI assistant for Insurellm, a company specializing in insurance technology and solutions.

I couldn't find relevant information in the knowledge base for this query. 

Please acknowledge this limitation and either:
1. Ask clarifying questions to help find the right information
2. Suggest related topics that might be helpful
3. Provide general guidance if appropriate (but clearly indicate it's not from the knowledge base)

Be helpful while being transparent about the lack of specific context."""


# Initialize vector store and LLM
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
llm = ChatOpenAI(temperature=0, model_name=MODEL)


def load_bm25_index():
    """Load BM25 index if it exists"""
    if os.path.exists(BM25_DB_NAME):
        with open(BM25_DB_NAME, 'rb') as f:
            return pickle.load(f)
    return None


vector_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={
        "score_threshold": 0.5, 
        "k": RETRIEVAL_K * 2
    })

def tokenize(text: str) -> List[str]:
    """Simple preprocessing: remove special chars and filter short tokens"""
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 2]

# Load BM25 documents
bm25_data = load_bm25_index()
if bm25_data:
    bm25_retriever = BM25Retriever.from_documents(
        bm25_data['chunks'],
        k=RETRIEVAL_K * 2,
        preprocess_func=tokenize
    )
    
    # Create ensemble retriever with RRF fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.75, 0.25],
        c=60
    )
else:
    ensemble_retriever = vector_retriever

_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def fetch_context(question: str) -> List[Document]:
    """
    Retrieve relevant context documents using ensemble retriever.
    """
    all_docs = ensemble_retriever.invoke(question)
    reranker = get_reranker()
    pairs = [[question, doc.page_content] for doc in all_docs]
    scores = reranker.predict(pairs)
    
    # Sort by score and return top FINAL_K
    doc_scores = list(zip(all_docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in doc_scores[:RETRIEVAL_K]]
    
    return top_docs



def format_context_with_metadata(docs: List[Document]) -> Tuple[str, str]:
    """
    Format documents into context string and metadata summary
    
    Returns:
        Tuple of (formatted_context, metadata_summary)
    """
    if not docs:
        return "", "No relevant context found."
    
    context_parts = []
    metadata_parts = []
    doc_types = set()
    
    for i, doc in enumerate(docs, 1):
        doc_type = doc.metadata.get('doc_type', 'unknown')
        is_parent = doc.metadata.get('is_parent', False)
        source = doc.metadata.get('source', 'unknown')
        
        doc_types.add(doc_type)
        
        # Format context with clear separation
        label = f"[{doc_type.upper()} - {'OVERVIEW' if is_parent else 'DETAIL'}]"
        context_parts.append(f"{label}\n{doc.page_content}\n")
        
        # Collect metadata
        metadata_parts.append(
            f"  {i}. Type: {doc_type}, Level: {'Parent' if is_parent else 'Child'}, Source: {Path(source).name if source != 'unknown' else 'N/A'}"
        )
    
    context = "\n".join(context_parts)
    metadata = f"Retrieved {len(docs)} documents from: {', '.join(sorted(doc_types))}\n" + "\n".join(metadata_parts)
    
    return context, metadata


def answer_question(
    question: str, 
    history: List[dict] = None,
    include_metadata: bool = True
) -> Tuple[str, List[Document]]:
    """
    Answer the given question using RAG with hybrid search.
    
    Args:
        question: User's question
        history: Conversation history
        include_metadata: Whether to include metadata in the prompt
    
    Returns:
        Tuple of (answer, context_documents)
    """
    if history is None:
        history = []
    
    # Fetch context using hybrid search with hierarchical awareness
    docs = fetch_context(question)
    
    # Format context and metadata
    context, metadata = format_context_with_metadata(docs)
    
    # Choose prompt based on whether context was found
    if docs:
        system_content = SYSTEM_PROMPT.format(
            schema=DATABASE_SCHEMA,
            context=context,
            metadata=metadata if include_metadata else "Context documents retrieved successfully."
        )
    else:
        system_content = NO_CONTEXT_PROMPT
    
    # Build message chain
    messages = [SystemMessage(content=system_content)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    
    llm_with_tools = llm.bind_tools([execute_sql_query])
    response = llm_with_tools.invoke(messages)
    if response.tool_calls:
        print(f"DEBUG: Tool calls detected: {len(response.tool_calls)}")
        messages.append(response)
        for tool_call in response.tool_calls:
            print(f"DEBUG: Tool call - name: {tool_call['name']}, id: {tool_call['id']}")
            try:
                if tool_call['name'] == 'execute_sql_query':
                    query = tool_call['args']['query']
                    reasoning = tool_call['args'].get('reasoning', '')
                    sql_result = execute_sql_query.invoke({"query": query, "reasoning": reasoning})
                    print(f"DEBUG: SQL_Result {sql_result}")
                    result_content = json.dumps(sql_result, indent=2)
                else:
                    # Handle unknown tool calls
                    result_content = json.dumps({"error": f"Unknown tool: {tool_call['name']}"})
                
                tool_message = ToolMessage(
                    content=result_content,
                    tool_call_id=tool_call['id']
                )
                messages.append(tool_message)
                
            except Exception as e:
                # Still add a ToolMessage even if execution fails
                messages.append(response)
                tool_message = ToolMessage(
                    content=json.dumps({"error": str(e)}),
                    tool_call_id=tool_call['id']
                )
                messages.append(tool_message)
        
        # Get final response after tool use
        final_response = llm.invoke(messages)
        return final_response.content, docs
    
    return response.content, docs




class SQLQueryInput(BaseModel):
    query: str = Field(description="The SQL query to execute. Should be a valid SQLite query.")
    reasoning: str = Field(description="Brief explanation of why this query answers the user's question.")

@tool(args_schema=SQLQueryInput)
def execute_sql_query(query: str, reasoning: str = "") -> dict:
    """
    Execute a SQL query and return results.
    
    Args:
        query: SQL query to execute
        DB_PATH: Path to SQLite database
        reasoning: Explanation of the query (optional)
    
    Returns:
        Dictionary with results or error
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute(query)
        
        # Handle SELECT queries
        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            # Convert to list of dicts
            results = [dict(row) for row in rows]
            conn.close()
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "query": query,
                "reasoning": reasoning
            }
        else:
            # Handle INSERT/UPDATE/DELETE
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return {
                "success": True,
                "affected_rows": affected,
                "query": query,
                "reasoning": reasoning
            }
            
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "query": query
        }


