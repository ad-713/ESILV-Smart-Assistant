import os
import shutil
import time
import concurrent.futures
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
CHROMA_PATH = "data/chroma_db"
DATA_PATH = "data/raw"

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})

def load_documents(files):
    """
    Loads PDF documents from the file paths.
    """
    documents = []
    for file in files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    """
    Splits documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List):
    total_start_time = time.time()
    
    print("--- Start add_to_chroma ---")
    
    # Measure DB Init
    t0 = time.time()
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )
    print(f"⏱️ DB Init: {time.time() - t0:.4f}s")

    # Measure ID calculation
    t0 = time.time()
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"⏱️ ID Calculation: {time.time() - t0:.4f}s")

    # Measure Fetching existing IDs
    t0 = time.time()
    # 1. Fetch existing IDs in one go (Optimized)
    existing_items = db.get(ids=[c.metadata["id"] for c in chunks_with_ids], include=[])
    existing_ids = set(existing_items["ids"])
    print(f"⏱️ Fetch Existing IDs: {time.time() - t0:.4f}s")
    
    # Measure Filtering
    t0 = time.time()
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]
    print(f"⏱️ Filtering New Chunks: {time.time() - t0:.4f}s")

    if not new_chunks:
        print("✅ No new documents to add")
        return

    print(f"👉 Adding {len(new_chunks)} new documents...")

    # 2. Use single-threaded, large batch ingestion
    BATCH_SIZE = 1024
    embedding_function = get_embedding_function()
    
    # Process in batches
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch_start_time = time.time()
        batch = new_chunks[i : i + BATCH_SIZE]
        
        # 3. Precompute embeddings ONCE
        embed_start_time = time.time()
        embeddings = embedding_function.embed_documents(
            [chunk.page_content for chunk in batch]
        )
        print(f"   ⏱️ Batch {i//BATCH_SIZE + 1} Embedding Generation: {time.time() - embed_start_time:.4f}s")
        
        add_start_time = time.time()
        db._collection.add(
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[c.metadata["id"] for c in batch],
            embeddings=embeddings
        )
        print(f"   ⏱️ Batch {i//BATCH_SIZE + 1} Add to DB: {time.time() - add_start_time:.4f}s")
        
        print(f"   Processed batch {i//BATCH_SIZE + 1}/{(len(new_chunks)-1)//BATCH_SIZE + 1} ({len(batch)} docs) in {time.time() - batch_start_time:.4f}s")

    print(f"✅ Finished in {time.time() - total_start_time:.2f}s")

def calculate_chunk_ids(chunks):
    """
    Create chunk IDs like "data/monopoly.pdf:6:2"
    Page Source : Page Number : Chunk Index
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def query_rag(query_text: str):
    """
    Query the RAG system and return the most relevant context.
    """
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
