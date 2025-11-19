# Advanced Local RAG System

A **production-ready** Retrieval-Augmented Generation (RAG) system with intelligent file tracking, content-based deduplication, and query abstraction.

## 🎯 Overview

This enterprise-grade RAG system features:
- **Local Embeddings**: HuggingFace `all-MiniLM-L6-v2` (384 dimensions)
- **Local LLM**: Ollama `llama3.2` 
- **Vector Database**: MongoDB Atlas Vector Search
- **Smart File Tracking**: Content-based hash persistence
- **Query Abstraction**: File-based query processing
- **Duplicate Detection**: Automatic content deduplication
- **Framework**: LangChain with optimized imports

**Key Benefits**: 100% local, intelligent file management, production-ready, cost-free.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **MongoDB** (local or Atlas)
- **Ollama** with llama3.2 model

### 2. Installation
```bash
# Clone and setup
git clone <your-repo>
cd personal-ai

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your MongoDB Atlas connection string
```

### 3. Setup Ollama
```bash
# Install Ollama: https://ollama.ai/
ollama pull llama3.2
ollama serve

# Verify installation
ollama run llama3.2 "Hello, how are you?"
```

### 4. Setup MongoDB Atlas Vector Index
Create a vector search index named `vector_index` in your Atlas cluster:
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding", 
    "numDimensions": 384,
    "similarity": "cosine"
  }]
}
```

### 5. Add Your Documents
```bash
# Place your documents in the docs/ directory
mkdir docs
# Copy your PDF, TXT, or MD files to docs/
```

### 6. Run the System
```bash
# Process documents and run queries
python example_usage.py

# Test content hashing features
python test_content_hashing.py
```

## 📖 Usage

### Smart Document Processing
```python
from rag_system import MongoRAGSystem

# Initialize with MongoDB Atlas
rag = MongoRAGSystem(
    mongodb_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="rag_database",
    collection_name="documents",
    ollama_model="llama3.2",
    embedding_model_name="all-MiniLM-L6-v2"
)

# Load documents (automatically skips already processed files)
documents = rag.load_documents_from_directory("./docs")
split_docs = rag.split_documents(documents)
rag.add_documents(split_docs)

# Query the system
result = rag.query("What are Zurik and Zorba?")
print(result["answer"])
print(f"Sources: {len(result['source_documents'])}")
```

### Query from File
```python
# Create queries.txt file with your questions
# Process all queries and save results
results = rag.process_queries_from_file("queries.txt", "results.json")
```

### Content Hash Management
```python
# Check if content is already processed (filename-independent)
if rag.is_content_already_processed("document.pdf"):
    print("Content already processed, skipping...")

# Get processing statistics
stats = rag.get_content_processing_stats()
print(f"Processed {stats['total_processed_files']} unique files")

# Find duplicate content
duplicates = rag.find_duplicate_content()
for dup in duplicates:
    print(f"Found {dup['count']} copies of same content")
```

## ⚙️ Configuration

### Environment Variables
```bash
# MongoDB Atlas (recommended)
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/database

# Ollama Configuration
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### System Parameters
- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Retrieval Count**: 4 documents (configurable)
- **Embedding Dimensions**: 384 (all-MiniLM-L6-v2)
- **Hash Algorithm**: SHA-256 for content tracking

## 🔧 API Reference

### MongoRAGSystem Constructor
```python
MongoRAGSystem(
    mongodb_uri: str,
    database_name: str, 
    collection_name: str,
    index_name: str = "vector_index",
    ollama_model: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434",
    embedding_model_name: str = "all-MiniLM-L6-v2"
)
```

### Document Processing Methods
- `load_documents_from_directory(path, glob="**/*.txt")` - Load documents from folder
- `load_documents_from_file(path)` - Load single document
- `split_documents(docs, chunk_size=1000, overlap=200)` - Split into chunks
- `add_documents(docs)` - Add to vector store

### Query Methods
- `query(question)` - Ask questions with RAG
- `similarity_search(query, k=4)` - Find similar documents
- `load_queries_from_file(file_path)` - Load queries from text file
- `process_queries_from_file(input_file, output_file)` - Batch process queries

### Content Hash Methods
- `is_content_already_processed(file_path)` - Check if content processed
- `mark_content_as_processed(file_path, doc_count, chunk_count)` - Mark as processed
- `get_content_processing_stats()` - Get processing statistics
- `find_duplicate_content()` - Find files with identical content
- `cleanup_orphaned_hashes()` - Remove orphaned hash records

## 🛠️ Troubleshooting

### MongoDB Issues
- **Connection Error**: Check MongoDB is running
- **Index Not Found**: Create vector search index in Atlas
- **Local Setup**: `brew install mongodb-community && brew services start mongodb-community`

### Ollama Issues  
- **Connection Error**: Run `ollama serve`
- **Model Missing**: Run `ollama pull llama3.2`
- **Test**: `ollama run llama3.2 "Hello"`

### Embedding Issues
- **First Run**: Model downloads automatically (~90MB)
- **Slow Loading**: Model caches locally after first use
- **GPU**: Change `device: 'cpu'` to `device: 'cuda'` if available

## 🏗️ Advanced Features

### 🔍 Intelligent File Tracking
- **Content-based hashing**: SHA-256 of file content (filename-independent)
- **Persistent storage**: Hash records stored in dedicated MongoDB collection
- **Duplicate detection**: Automatically identifies identical content
- **Orphan cleanup**: Removes records for deleted files
- **Statistics**: Comprehensive analytics on processed content

### 📝 Query Abstraction
- **File-based queries**: Load questions from `queries.txt`
- **Batch processing**: Process multiple queries automatically
- **Result export**: Save answers to JSON format
- **Comment support**: Use `#` for comments in query files

### 🚀 Performance Optimizations
- **Skip processed files**: Avoid re-processing unchanged content
- **Indexed collections**: Optimized database queries
- **Chunked processing**: Efficient memory usage
- **Deprecation-free**: Updated imports and methods

## 📁 Project Structure
```
personal-ai/
├── rag_system.py           # Main RAG implementation with advanced features
├── example_usage.py        # Complete workflow example
├── test_content_hashing.py # Content hash system testing
├── queries.txt            # Query input file
├── query_results.json     # Query output file (generated)
├── requirements.txt       # Minimal dependencies
├── .env.example           # Environment template
├── docs/                  # Document directory
└── README.md             # This documentation
```

## 📊 MongoDB Collections

### `documents` Collection
Stores document chunks with embeddings for vector search.

### `content_hashes` Collection
Persistent file tracking with content-based hashing:
```json
{
  "content_hash": "sha256_hash_of_content",
  "current_file_path": "/path/to/file.pdf",
  "file_size": 1638740,
  "processed_time": "2025-11-19T...",
  "document_count": 3,
  "chunk_count": 10,
  "status": "processed"
}
```

## 🔒 Privacy & Security
- **100% Local Processing**: No external API calls for embeddings or LLM
- **Data Privacy**: Your documents never leave your infrastructure
- **Content Security**: SHA-256 hashing for integrity verification
- **No API Keys**: Completely self-contained system
- **Cost-Free**: No per-token or usage charges

## 🎯 Production Ready
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed status and progress reporting
- **Scalability**: Efficient MongoDB Atlas integration
- **Maintainability**: Clean, documented codebase
- **Testing**: Validation scripts included

## 📄 License
MIT License
