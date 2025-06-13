import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Use your existing imports to avoid issues
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

def main():
    print("Simple Database Population")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Load documents
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Add to database
    print("Adding to database...")
    add_to_chroma(chunks)
    
    # Verify
    print("Verifying database...")
    verify_database()

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Initialize database
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing documents
    try:
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Existing documents: {len(existing_ids)}")
    except:
        existing_ids = set()
        print("New database")

    # Filter new chunks
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) == 0:
        print("No new documents to add")
        return

    print(f"Adding {len(new_chunks)} new documents...")
    
    # Add in smaller batches to avoid issues
    batch_size = 50
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        
        print(f"   Adding batch {i//batch_size + 1} ({len(batch)} docs)...")
        
        try:
            db.add_documents(batch, ids=batch_ids)
            print(f"   Batch {i//batch_size + 1} added successfully")
        except Exception as e:
            print(f"   Batch {i//batch_size + 1} failed: {e}")
            continue

    print("Database population complete!")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def verify_database():
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=get_embedding_function()
        )
        
        all_docs = db.get(include=[])
        doc_count = len(all_docs["ids"])
        print(f"Database contains: {doc_count} documents")
        
        if doc_count > 0:
            # Test query
            results = db.similarity_search("Nepal inflation 2022", k=3)
            print(f"Test search found: {len(results)} results")
            
            if results:
                print("Database working correctly!")
                print("\nReady to query! Try:")
                print('python query_data.py "What is the annual average consumer price inflation in Nepal in year 2022-2023?"')
            else:
                print("Database populated but search not working")
        else:
            print("Database is empty")
            
    except Exception as e:
        print(f"Verification error: {e}")

if __name__ == "__main__":
    main()