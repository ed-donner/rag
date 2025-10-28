import os
import glob
import re
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

MODEL = "thenlper/gte-small"

DB_NAME = str(Path(__file__).parent.parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent.parent / "knowledge-base")

embeddings = HuggingFaceEmbeddings(model_name=MODEL)

load_dotenv(override=True)


def extract_metadata(doc, folder):
    """Extract extra metadata from document"""
    metadata = doc.metadata.copy()
    doc_type = os.path.basename(folder)
    
    # Get entity name from filename
    filename = Path(doc.metadata['source']).stem
    metadata['entity_name'] = filename
    metadata['doc_type'] = doc_type
    
    # Parse document for title
    lines = doc.page_content.split('\n')
    metadata['title'] = lines[0].replace('#', '').strip() if lines else ''
    
    # Entity-specific extraction
    if doc_type == 'employees':
        # Extract: name, title, salary, location, dob
        salary_match = re.search(r'\*\*Current Salary:\*\*\s*(\$[\d,]+)', doc.page_content)
        if salary_match:
            metadata['salary'] = salary_match.group()
        
        title_match = re.search(r'\*\*Job Title:\*\*\s*(.+)', doc.page_content)
        if title_match:
            metadata['job_title'] = title_match.group(1).strip()
        
        location_match = re.search(r'\*\*Location:\*\*\s*(.+)', doc.page_content)
        if location_match:
            metadata['location'] = location_match.group(1).strip()
        
        dob_match = re.search(r'\*\*Date of Birth:\*\*\s*(.+)', doc.page_content)
        if dob_match:
            metadata['dob'] = dob_match.group(1).strip()
    elif doc_type == 'contracts':
        # Extract: contract number, client, product, monthly cost
        contract_num = re.search(r'\*\*Contract [Number|ID]:\*\*\s*(.+)', doc.page_content)
        if contract_num:
            metadata['contract_number'] = contract_num.group(1).strip()
        
        monthly_cost = re.search(r'[monthly payments? of |](\$[\d,]+)[|\sper month]', doc.page_content, re.IGNORECASE)
        if monthly_cost:
            metadata['monthly_payment'] = monthly_cost.group()
        
        # Extract client and product from filename
        if 'Contract with' in filename:
            parts = filename.split(' for ')
            if len(parts) > 0:
                metadata['client_name'] = parts[0].replace('Contract with ', '')
            if len(parts) > 1:
                metadata['product_name'] = parts[1]
    
    elif doc_type == 'products':
        metadata['product_name'] = filename
        
        # Extract pricing tiers
        pricing_section = re.search(r'## Pricing(.+?)(?=##|\Z)', doc.page_content, re.DOTALL)
        if pricing_section:
            tier_prices = re.findall(r'\$[\d,]+/month', pricing_section.group(1))
            if tier_prices:
                metadata['pricing_info'] = ', '.join(tier_prices)
    
    elif doc_type == 'company':
        metadata['company_doc'] = filename
    
    return metadata


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            # Apply metadata extraction
            doc.metadata = extract_metadata(doc, folder)
            documents.append(doc)
    return documents


def create_chunks(documents):
    # Optimized chunking based on knowledge base analysis:
    # - Avg file size: 4004 chars
    # - Refined tuning shows 2000 is best with GTE-small
    # - 2000 chunks: files break into 1-2 pieces (maximum context preservation)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,  # 20% of chunk_size
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # Prefer splitting on headers
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    print(f"Loading documents from {KNOWLEDGE_BASE}...")
    documents = fetch_documents()
    print(f"Loaded {len(documents)} documents")
    
    print("Creating chunks...")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings and vector store...")
    create_embeddings(chunks)
    print("Ingestion complete")
