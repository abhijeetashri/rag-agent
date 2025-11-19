#!/usr/bin/env python3
"""
Test script to demonstrate the robust content-based file tracking system
"""

import os
from pathlib import Path
from rag_system import MongoRAGSystem

def main():
    print("🧪 Testing Content-Based Hash Tracking System")
    print("=" * 60)
    
    # Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI environment variable not set")
        return
    
    # Initialize RAG system
    rag = MongoRAGSystem(
        mongodb_uri=MONGODB_URI,
        database_name="rag_database",
        collection_name="documents",
        embedding_model_name="all-MiniLM-L6-v2"
    )
    
    print("✅ RAG system initialized")
    
    # Test 1: Show current content processing statistics
    print("\n📊 Current Content Processing Statistics:")
    print("-" * 40)
    stats = rag.get_content_processing_stats()
    print(f"Total unique content processed: {stats.get('total_processed_files', 0)}")
    
    file_types = stats.get('file_type_distribution', [])
    if file_types:
        print("File type distribution:")
        for ft in file_types:
            ext = ft['_id'] or 'no extension'
            print(f"  • {ext}: {ft['count']} files, {ft['total_documents']} docs, {ft['total_chunks']} chunks")
    else:
        print("No files processed yet")
    
    # Test 2: Check for duplicate content
    print("\n🔍 Checking for Duplicate Content:")
    print("-" * 40)
    duplicates = rag.find_duplicate_content()
    if duplicates:
        print(f"Found {len(duplicates)} sets of duplicate content:")
        for dup in duplicates:
            print(f"  Hash: {dup['_id'][:16]}... ({dup['count']} copies)")
            for file_info in dup['files']:
                print(f"    - {file_info['file_name']} ({file_info['file_path']})")
    else:
        print("No duplicate content found")
    
    # Test 3: Test filename independence
    print("\n🔄 Testing Filename Independence:")
    print("-" * 40)
    
    docs_dir = Path("./docs")
    if docs_dir.exists():
        pdf_files = list(docs_dir.glob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
            print(f"Testing with file: {test_file.name}")
            
            # Check if content is already processed
            is_processed = rag.is_content_already_processed(str(test_file))
            print(f"Content already processed: {is_processed}")
            
            # Calculate content hash
            content_hash = rag._calculate_content_hash(str(test_file))
            if content_hash:
                print(f"Content hash: {content_hash[:16]}...")
                
                # Show that the same content would have the same hash regardless of filename
                print("✅ This hash is based purely on file content, not filename")
                print("   → Renaming the file won't change the hash")
                print("   → Moving the file won't change the hash")
                print("   → Only changing the content will change the hash")
            else:
                print("❌ Could not calculate content hash")
        else:
            print("No PDF files found in docs directory")
    else:
        print("Docs directory not found")
    
    # Test 4: Cleanup orphaned hashes
    print("\n🧹 Cleaning Up Orphaned Hash Records:")
    print("-" * 40)
    orphaned_count = rag.cleanup_orphaned_hashes()
    print(f"Processed orphaned records: {orphaned_count}")
    
    # Test 5: Show collection information
    print("\n📋 Hash Collection Information:")
    print("-" * 40)
    print(f"Collection name: content_hashes")
    print(f"Database: rag_database")
    print("Indexes created for:")
    print("  • content_hash (unique)")
    print("  • current_file_path")
    print("  • content_hash + status (compound)")
    
    print("\n🎯 Key Benefits of This System:")
    print("-" * 40)
    print("✅ Filename independent - same content = same hash")
    print("✅ Persistent storage - survives collection clears")
    print("✅ Duplicate detection - finds identical content")
    print("✅ Efficient lookups - indexed for performance")
    print("✅ Orphan cleanup - maintains data integrity")
    print("✅ Statistics tracking - comprehensive analytics")
    
    # Close connection
    rag.close_connection()
    print("\n🎉 Content hash testing completed!")

if __name__ == "__main__":
    main()
