import os
import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def check_data_directory():
    """Check if data directory exists and contains PDF files"""
    print("=== CHECKING DATA DIRECTORY ===")
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data directory '{DATA_PATH}' does not exist!")
        return False
    
    pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
    print(f"Data directory exists: {DATA_PATH}")
    print(f"PDF files found: {len(pdf_files)}")
    
    for pdf in pdf_files:
        print(f"   - {pdf.name} ({pdf.stat().st_size} bytes)")
    
    if not pdf_files:
        print("ERROR: No PDF files found in data directory!")
        return False
    
    return True

def check_chroma_database():
    """Check if Chroma database exists and contains documents"""
    print("\n=== CHECKING CHROMA DATABASE ===")
    
    if not os.path.exists(CHROMA_PATH):
        print(f"ERROR: Chroma directory '{CHROMA_PATH}' does not exist!")
        print("Run: python populate_database.py")
        return False
    
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get all documents
        existing_items = db.get(include=["documents", "metadatas"])
        doc_count = len(existing_items["ids"])
        
        print(f"Chroma database exists: {CHROMA_PATH}")
        print(f"Documents in database: {doc_count}")
        
        if doc_count == 0:
            print("ERROR: Database is empty!")
            print("Run: python populate_database.py")
            return False
        
        # Show sample documents
        print("\n Sample document IDs:")
        for i, doc_id in enumerate(existing_items["ids"][:3]):
            print(f"   {i+1}. {doc_id}")
        
        if doc_count > 3:
            print(f"   ... and {doc_count - 3} more")
            
        return True
        
    except Exception as e:
        print(f"ERROR accessing database: {e}")
        return False

def test_document_loading():
    """Test loading documents from PDF files"""
    print("\n=== TESTING DOCUMENT LOADING ===")
    
    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = document_loader.load()
        
        print(f"Documents loaded: {len(documents)}")
        
        if documents:
            # Show sample content
            sample_doc = documents[0]
            print(f"\n Sample document metadata:")
            print(f"   Source: {sample_doc.metadata.get('source', 'Unknown')}")
            print(f"   Page: {sample_doc.metadata.get('page', 'Unknown')}")
            print(f"   Content length: {len(sample_doc.page_content)} characters")
            print(f"\n First 200 characters of content:")
            print(f"   {sample_doc.page_content[:200]}...")
            
            # Check for Nepal-related content
            nepal_mentions = sum(1 for doc in documents if 'nepal' in doc.page_content.lower())
            inflation_mentions = sum(1 for doc in documents if 'inflation' in doc.page_content.lower())
            
            print(f"\n Content analysis:")
            print(f"   Documents mentioning 'Nepal': {nepal_mentions}")
            print(f"   Documents mentioning 'inflation': {inflation_mentions}")
            
        return len(documents) > 0
        
    except Exception as e:
        print(f" ERROR loading documents: {e}")
        return False

def test_similarity_search():
    """Test similarity search with the problematic query"""
    print("\n=== TESTING SIMILARITY SEARCH ===")
    
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        test_queries = [
            "What is the annual average consumer price inflation in Nepal in year 2022-2023?",
            "Nepal inflation",
            "consumer price",
            "2022-2023"
        ]
        
        for query in test_queries:
            print(f"\n Testing query: '{query}'")
            results = db.similarity_search_with_score(query, k=3)
            print(f"   Results found: {len(results)}")
            
            for i, (doc, score) in enumerate(results):
                print(f"   {i+1}. Score: {score:.4f}")
                print(f"      Source: {doc.metadata.get('id', 'Unknown')}")
                print(f"      Content preview: {doc.page_content[:100]}...")
                
    except Exception as e:
        print(f" ERROR in similarity search: {e}")

def test_embedding_function():
    """Test if embedding function works correctly"""
    print("\n=== TESTING EMBEDDING FUNCTION ===")
    
    try:
        embedding_function = get_embedding_function()
        
        # Test embedding a simple text
        test_text = "Nepal consumer price inflation 2022-2023"
        embeddings = embedding_function.embed_query(test_text)
        
        print(f"   Embedding function works")
        print(f"   Embedding dimension: {len(embeddings)}")
        print(f"   Sample values: {embeddings[:5]}")
        
        return True
        
    except Exception as e:
        print(f" ERROR with embedding function: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print(" RAG SYSTEM DIAGNOSTIC\n")
    
    checks = [
        ("Data Directory", check_data_directory),
        ("Document Loading", test_document_loading),
        ("Embedding Function", test_embedding_function),
        ("Chroma Database", check_chroma_database),
        ("Similarity Search", test_similarity_search)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"CRITICAL ERROR in {name}: {e}")
            results[name] = False
    
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:20} {status}")
    
    print("\nRECOMMENDATIONS:")
    
    if not results.get("Data Directory"):
        print("1. Create 'data' folder and add your PDF files")
    
    if not results.get("Chroma Database"):
        print("2. Run: python populate_database.py")
        print("3. If database exists but empty, run: python populate_database.py --reset")
    
    if not results.get("Embedding Function"):
        print("4. Check if Ollama is running: ollama serve")
        print("5. Verify nomic-embed-text model: ollama pull nomic-embed-text")

if __name__ == "__main__":
    main()