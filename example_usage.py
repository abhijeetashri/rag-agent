#!/usr/bin/env python3
"""
Complete RAG system implementation with MongoDB Atlas
Flow:
1. Create vector index in MongoDB Atlas
2. Load documents from 'docs' directory
3. Store chunks in 'documents' collection in 'rag_database' database
4. Process queries from queries.txt file
"""

import os
from pathlib import Path
from rag_system import MongoRAGSystem

def main():
    print("🚀 Starting Complete RAG System Flow")
    print("=" * 60)
    
    # Configuration - using MongoDB Atlas
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI environment variable not set")
        print("Please copy .env.example to .env and configure your MongoDB Atlas connection")
        return
    
    DATABASE_NAME = "rag_database"
    COLLECTION_NAME = "documents"
    INDEX_NAME = "vector_index"
    DOCS_DIR = "./docs"
    
    # Step 1: Initialize the RAG system
    print("\n📡 Step 1: Initializing RAG system with MongoDB Atlas...")
    try:
        rag = MongoRAGSystem(
            mongodb_uri=MONGODB_URI,
            database_name=DATABASE_NAME,
            collection_name=COLLECTION_NAME,
            index_name=INDEX_NAME,
            ollama_model="llama3.2",
            ollama_base_url="http://localhost:11434",
            embedding_model_name="all-MiniLM-L6-v2"
        )
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Step 2: Create vector index (you mentioned it's already created, but we'll call it anyway)
    print("\n🔍 Step 2: Setting up vector index...")
    try:
        rag.create_vector_index()
        print("✅ Vector index setup completed")
    except Exception as e:
        print(f"⚠️  Vector index setup: {e}")
        print("Continuing with existing index...")
    
    # Step 3: Load documents from docs directory
    print(f"\n📂 Step 3: Loading documents from '{DOCS_DIR}' directory...")
    docs_path = Path(DOCS_DIR)
    
    if not docs_path.exists():
        print(f"❌ Error: '{DOCS_DIR}' directory not found")
        return
    
    # Check what files we have
    files = list(docs_path.glob("*"))
    print(f"Found {len(files)} files in docs directory:")
    for file in files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    # Load documents based on file types with file tracking
    all_documents = []
    new_files_processed = 0
    
    for file_path in files:
        # Skip non-document files
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"📸 Skipping image file: {file_path.name} (image processing not implemented)")
            continue
        elif file_path.suffix.lower() not in ['.pdf', '.txt', '.md']:
            print(f"❓ Skipping unknown file type: {file_path.name}")
            continue
        
        # Check if content is already processed (filename-independent)
        if rag.is_content_already_processed(str(file_path)):
            print(f"⏭️  Skipping already processed content: {file_path.name}")
            continue
        
        # Process new or changed files
        if file_path.suffix.lower() == '.pdf':
            print(f"\n📄 Loading PDF: {file_path.name}")
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                print(f"✅ Loaded {len(docs)} pages from PDF")
                all_documents.extend(docs)
                rag.mark_content_as_processed(str(file_path), len(docs), 0)
                new_files_processed += 1
            except Exception as e:
                print(f"❌ Error loading PDF {file_path.name}: {e}")
                
        elif file_path.suffix.lower() in ['.txt', '.md']:
            print(f"\n📝 Loading text file: {file_path.name}")
            try:
                docs = rag.load_documents_from_file(str(file_path))
                print(f"✅ Loaded text file")
                all_documents.extend(docs)
                rag.mark_content_as_processed(str(file_path), len(docs), 0)
                new_files_processed += 1
            except Exception as e:
                print(f"❌ Error loading text file {file_path.name}: {e}")
    
    print(f"\n📊 File Processing Summary:")
    print(f"   - New files processed: {new_files_processed}")
    print(f"   - Total documents from new files: {len(all_documents)}")
    
    # Show content processing statistics
    content_stats = rag.get_content_processing_stats()
    print(f"   - Total unique content processed: {content_stats.get('total_processed_files', 0)}")
    
    # Show file type distribution
    file_types = content_stats.get('file_type_distribution', [])
    if file_types:
        print("   - File type distribution:")
        for ft in file_types:
            ext = ft['_id'] or 'no extension'
            print(f"     • {ext}: {ft['count']} files, {ft['total_documents']} docs, {ft['total_chunks']} chunks")
    
    if not all_documents and new_files_processed == 0:
        print("ℹ️  No new files to process. All files are already up to date.")
        print("   Proceeding with existing documents in database...")
    elif not all_documents:
        print("❌ No documents were loaded successfully")
        return
    
    # Step 4: Split documents into chunks (only for new documents)
    if all_documents:
        print("\n✂️  Step 4: Splitting documents into chunks...")
        try:
            split_docs = rag.split_documents(all_documents, chunk_size=1000, chunk_overlap=200)
            print(f"✅ Created {len(split_docs)} document chunks")
        except Exception as e:
            print(f"❌ Error splitting documents: {e}")
            return
        
        # Step 5: Store chunks in MongoDB Atlas
        print(f"\n💾 Step 5: Storing new chunks in MongoDB Atlas...")
        print(f"Database: {DATABASE_NAME}")
        print(f"Collection: {COLLECTION_NAME}")
        
        try:
            # Add only new documents (don't clear existing ones)
            existing_count = rag.collection.count_documents({})
            print(f"Found {existing_count} existing documents in collection")
            
            # Add new documents
            doc_ids = rag.add_documents(split_docs)
            print(f"✅ Successfully added {len(doc_ids)} new document chunks to MongoDB Atlas")
            
            # Update content hash records with actual chunk counts
            # Group chunks by source file to update hash records
            chunk_counts_by_file = {}
            for doc in split_docs:
                source = doc.metadata.get('source', 'unknown')
                chunk_counts_by_file[source] = chunk_counts_by_file.get(source, 0) + 1
            
            # Update hash records with chunk counts
            for file_path, chunk_count in chunk_counts_by_file.items():
                if Path(file_path).exists():
                    rag.mark_content_as_processed(file_path, 0, chunk_count)
            
            # Verify storage
            final_count = rag.collection.count_documents({})
            print(f"✅ Total documents in collection: {final_count}")
            
        except Exception as e:
            print(f"❌ Error storing documents: {e}")
            print("Make sure your MongoDB Atlas connection is working and vector index is created")
            return
    else:
        print("\n⏭️  Step 4-5: Skipping document processing (no new documents)")
        existing_count = rag.collection.count_documents({})
        print(f"Using existing {existing_count} documents in collection")
    
    # Step 6: Setup retrieval chain
    print("\n🔗 Step 6: Setting up retrieval chain...")
    try:
        rag.setup_retrieval_chain(k=4)
        print("✅ Retrieval chain configured")
    except Exception as e:
        print(f"❌ Error setting up retrieval chain: {e}")
        return
    
    # Step 7: Process queries from file
    print("\n❓ Step 7: Processing queries from file...")
    print("=" * 60)
    
    queries_file = "./queries.txt"
    results_file = "./query_results.json"
    
    if Path(queries_file).exists():
        print(f"📝 Using queries from: {queries_file}")
        try:
            # Process all queries from file and save results
            results = rag.process_queries_from_file(queries_file, results_file)
            print(f"\n📊 Query Processing Summary:")
            print(f"   - Total queries processed: {len(results)}")
            successful_queries = [r for r in results if 'error' not in r]
            print(f"   - Successful queries: {len(successful_queries)}")
            print(f"   - Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Error processing queries from file: {e}")
    else:
        print(f"⚠️  Queries file not found: {queries_file}")
        print("Creating sample queries file...")
        
        # Create sample queries file
        sample_queries = [
            "What is the main topic of the document?",
            "Summarize the key findings",
            "What equipment is used in the process?",
            "What are the benefits of this process?",
            "How does the separation process work?"
        ]
        
        try:
            with open(queries_file, 'w', encoding='utf-8') as f:
                f.write("# Sample queries for RAG system\n")
                f.write("# Lines starting with # are comments\n")
                f.write("# One query per line\n\n")
                for query in sample_queries:
                    f.write(f"{query}\n")
            
            print(f"✅ Created sample queries file: {queries_file}")
            print("You can edit this file to add your own queries and run the script again.")
            
        except Exception as e:
            print(f"❌ Error creating queries file: {e}")
    
    # Close connection
    print("\n🔚 Closing MongoDB connection...")
    rag.close_connection()
    
    print("\n🎉 RAG system flow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
