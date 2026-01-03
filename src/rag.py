import os
import shutil
import time
import concurrent.futures
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Configuration
CHROMA_PATH = "data/chroma_db"
DATA_PATH = "data/raw"

def get_embedding_function():
    return OllamaEmbeddings(model="qwen3-embedding:0.6b")

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
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    # 1. Fetch existing IDs in one go
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if not new_chunks:
        print("✅ No new documents to add")
        return

    print(f"👉 Adding {len(new_chunks)} new documents...")

    # 2. Optimized Batch Size (250 is usually ideal for local GPUs)
    BATCH_SIZE = 250 
    start_time = time.time()

    # Get parallel execution count from env
    num_parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", 1))

    # Helper function to process a batch
    def process_batch(batch):
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        db.add_documents(batch, ids=batch_ids)
        return len(batch)

    # Prepare batches
    batches = [new_chunks[i : i + BATCH_SIZE] for i in range(0, len(new_chunks), BATCH_SIZE)]
    
    print(f"🚀 Processing with {num_parallel} parallel threads")
    
    total_processed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                count = future.result()
                total_processed += count
                elapsed = time.time() - start_time
                avg_speed = total_processed / elapsed
                print(f"   Done {total_processed}/{len(new_chunks)} | Speed: {avg_speed:.1f} docs/sec")
            except Exception as e:
                print(f"❌ Error processing batch: {e}")

    print(f"✅ Finished in {time.time() - start_time:.2f}s")

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
