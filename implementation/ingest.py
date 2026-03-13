import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "text-embedding-3-large"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 220

# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

load_dotenv(override=True)


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    # Enrich each chunk with doc_type context prefix
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "")
        source = chunk.metadata.get("source", "")
        filename = Path(source).stem if source else ""

        # Prepend context so the embedding captures it
        chunk.page_content = (
            f"[Document: {filename} | Category: {doc_type}]\n\n"
            + chunk.page_content
        )
    return chunks

# def create_chunks(documents):
#     # First split by markdown headers to preserve structure
#     md_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=[
#             ("#", "h1"), ("##", "h2"), ("###", "h3")
#         ],
#         strip_headers=False  # Keep headers in chunk for context
#     )

#     # Then apply recursive splitting for large sections
#     text_splitter = MarkdownTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     all_chunks = []
#     for doc in documents:
#         # Try markdown splitting first
#         try:
#             md_chunks = md_splitter.split_text(doc.page_content)
#             for md_chunk in md_chunks:
#                 # Further split if too large
#                 if len(md_chunk.page_content) > CHUNK_SIZE:
#                     sub_chunks = text_splitter.create_documents(
#                         [md_chunk.page_content],
#                         metadatas=[{**doc.metadata, **md_chunk.metadata}]
#                     )
#                     all_chunks.extend(sub_chunks)
#                 else:
#                     md_chunk.metadata.update(doc.metadata)
#                     all_chunks.append(md_chunk)
#         except:
#             # Fallback to regular splitting
#             chunks = text_splitter.split_documents([doc])
#             all_chunks.extend(chunks)

#     return all_chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME,
               embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])[
        "embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
