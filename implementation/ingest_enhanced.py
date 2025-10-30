import os
import glob
import re
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

# NFL team's optimal embedding model
MODEL = "thenlper/gte-small"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = HuggingFaceEmbeddings(model_name=MODEL)

load_dotenv(override=True)


def extract_enhanced_metadata(doc, folder):
    """
    Enhanced metadata extraction combining NFL team's approach with our improvements.
    """
    metadata = doc.metadata.copy()
    doc_type = os.path.basename(folder)
    
    # Get entity name from filename
    filename = Path(doc.metadata['source']).stem
    metadata['entity_name'] = filename
    metadata['doc_type'] = doc_type
    
    # Parse document for title
    lines = doc.page_content.split('\n')
    metadata['title'] = lines[0].replace('#', '').strip() if lines else ''
    
    # Enhanced entity-specific extraction
    if doc_type == 'employees':
        # Extract: name, title, salary, location, dob, department
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
        
        # Extract department/team information
        dept_match = re.search(r'\*\*Department:\*\*\s*(.+)', doc.page_content)
        if dept_match:
            metadata['department'] = dept_match.group(1).strip()
        
        # Extract years of experience
        exp_match = re.search(r'(\d+)\s*years?\s*of\s*experience', doc.page_content, re.IGNORECASE)
        if exp_match:
            metadata['years_experience'] = exp_match.group(1)
        
        # Extract skills/technologies
        skills_match = re.search(r'\*\*Skills:\*\*\s*(.+)', doc.page_content)
        if skills_match:
            metadata['skills'] = skills_match.group(1).strip()
            
    elif doc_type == 'contracts':
        # Extract: contract number, client, product, monthly cost, duration
        contract_num = re.search(r'\*\*Contract [Number|ID]:\*\*\s*(.+)', doc.page_content)
        if contract_num:
            metadata['contract_number'] = contract_num.group(1).strip()
        
        monthly_cost = re.search(r'[monthly payments? of |](\$[\d,]+)[|\sper month]', doc.page_content, re.IGNORECASE)
        if monthly_cost:
            metadata['monthly_payment'] = monthly_cost.group()
        
        # Extract contract duration
        duration_match = re.search(r'(\d+)\s*(?:month|year)s?\s*(?:contract|term)', doc.page_content, re.IGNORECASE)
        if duration_match:
            metadata['contract_duration'] = duration_match.group(1) + " " + duration_match.group(2)
        
        # Extract contract status
        status_match = re.search(r'\*\*Status:\*\*\s*(.+)', doc.page_content)
        if status_match:
            metadata['contract_status'] = status_match.group(1).strip()
        
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
        
        # Extract product category
        category_match = re.search(r'\*\*Category:\*\*\s*(.+)', doc.page_content)
        if category_match:
            metadata['product_category'] = category_match.group(1).strip()
        
        # Extract target audience
        audience_match = re.search(r'\*\*Target Audience:\*\*\s*(.+)', doc.page_content)
        if audience_match:
            metadata['target_audience'] = audience_match.group(1).strip()
    
    elif doc_type == 'company':
        metadata['company_doc'] = filename
        
        # Extract company values/culture
        values_match = re.search(r'\*\*Values:\*\*\s*(.+)', doc.page_content)
        if values_match:
            metadata['company_values'] = values_match.group(1).strip()
        
        # Extract company size
        size_match = re.search(r'(\d+)\s*employees?', doc.page_content, re.IGNORECASE)
        if size_match:
            metadata['company_size'] = size_match.group(1)
    
    # Add content quality indicators
    content_length = len(doc.page_content)
    metadata['content_length'] = content_length
    metadata['has_structured_data'] = bool(re.search(r'\*\*.*\*\*', doc.page_content))
    metadata['has_numbers'] = bool(re.search(r'\d+', doc.page_content))
    metadata['has_dates'] = bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\b', doc.page_content))
    
    return metadata


def fetch_documents():
    """
    Load documents with enhanced metadata extraction.
    """
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            # Apply enhanced metadata extraction
            doc.metadata = extract_enhanced_metadata(doc, folder)
            documents.append(doc)
    return documents


def create_optimized_chunks(documents):
    """
    NFL team's optimal chunking strategy with our enhancements.
    """
    # NFL team's proven optimal settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # NFL team's optimal
        chunk_overlap=400,  # 20% of chunk_size
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # Header-aware
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk-level metadata for better retrieval
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_length'] = len(chunk.page_content)
        
        # Add content type indicators
        content = chunk.page_content.lower()
        chunk.metadata['contains_numbers'] = bool(re.search(r'\d+', content))
        chunk.metadata['contains_dates'] = bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\b', content))
        chunk.metadata['contains_money'] = bool(re.search(r'\$[\d,]+', content))
        chunk.metadata['contains_percentages'] = bool(re.search(r'\d+%', content))
        
        # Add semantic indicators
        chunk.metadata['is_employee_info'] = any(word in content for word in ['salary', 'job title', 'department', 'employee'])
        chunk.metadata['is_contract_info'] = any(word in content for word in ['contract', 'agreement', 'payment', 'client'])
        chunk.metadata['is_product_info'] = any(word in content for word in ['product', 'service', 'pricing', 'features'])
        chunk.metadata['is_company_info'] = any(word in content for word in ['company', 'founded', 'headquarters', 'mission'])
    
    return chunks


def create_embeddings(chunks):
    """
    Create vector store with GTE-small embeddings and enhanced metadata.
    """
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
    print(f"Enhanced metadata fields: {list(chunks[0].metadata.keys()) if chunks else 'None'}")
    
    return vectorstore


if __name__ == "__main__":
    print(f"Loading documents from {KNOWLEDGE_BASE}...")
    documents = fetch_documents()
    print(f"Loaded {len(documents)} documents")
    
    print("Creating optimized chunks...")
    chunks = create_optimized_chunks(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings and vector store...")
    create_embeddings(chunks)
    print("Enhanced ingestion complete")
