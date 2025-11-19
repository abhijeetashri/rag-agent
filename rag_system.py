import hashlib
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

load_dotenv()

class MongoRAGSystem:
    def __init__(
        self,
        mongodb_uri: str,
        database_name: str,
        collection_name: str,
        index_name: str = "vector_index",
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG system with MongoDB Atlas Vector Search
        
        Args:
            mongodb_uri: MongoDB connection string
            database_name: Name of the MongoDB database
            collection_name: Name of the collection to store documents
            index_name: Name of the vector search index
            ollama_model: Ollama model name (default: llama3.2)
            ollama_base_url: Ollama server URL (default: http://localhost:11434)
            embedding_model_name: HuggingFace embedding model name (default: all-MiniLM-L6-v2)
        """
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.index_name = index_name
        
        # Initialize MongoDB client
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Initialize embeddings with HuggingFace model
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        
        # Initialize Ollama LLM
        self.ollama_model = ollama_model
        self.llm = Ollama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7
        )
        
        # Initialize retrieval chain
        self.qa_chain = None
        
        # Content hash collection for persistent file tracking
        self.content_hash_collection = self.db["content_hashes"]

        # Create indexes for better performance
        self._setup_hash_collection_indexes()
    
    def load_documents_from_directory(self, directory_path: str, glob_pattern: str = "**/*.txt") -> List[Document]:
        """
        Load documents from a directory
        
        Args:
            directory_path: Path to the directory containing documents
            glob_pattern: Pattern to match files (default: all .txt files)
            
        Returns:
            List of Document objects
        """
        loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=TextLoader)
        documents = loader.load()
        return documents
    
    def load_documents_from_file(self, file_path: str) -> List[Document]:
        """
        Load documents from a single file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        return self.vector_store.add_documents(documents)
    
    def create_vector_index(self):
        """
        Create a vector search index in MongoDB Atlas
        Note: This requires Atlas Search to be configured in your MongoDB Atlas cluster
        """
        # Get embedding dimension based on model
        embedding_dim = 384 if "all-MiniLM-L6-v2" in self.embedding_model_name else 768
        
        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": embedding_dim,  # all-MiniLM-L6-v2 uses 384 dimensions
                    "similarity": "cosine"
                }
            ]
        }
        
        print(f"Please create a vector search index named '{self.index_name}' in your MongoDB Atlas cluster")
        print(f"Database: {self.database_name}")
        print(f"Collection: {self.collection_name}")
        print(f"Index definition: {index_definition}")
        print("\nYou can create this index through the MongoDB Atlas UI or using the Atlas CLI")
    
    def setup_retrieval_chain(self, k: int = 4):
        """
        Setup the retrieval QA chain
        
        Args:
            k: Number of documents to retrieve
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            self.setup_retrieval_chain()
        
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def _setup_hash_collection_indexes(self):
        """Setup indexes for the content hash collection for better performance"""
        try:
            # Create index on content_hash for fast lookups
            self.content_hash_collection.create_index("content_hash", unique=True)
            # Create index on file_path for file-based queries
            self.content_hash_collection.create_index("current_file_path")
            # Create compound index for efficient queries
            self.content_hash_collection.create_index([("content_hash", 1), ("status", 1)])
        except Exception as e:
            # Indexes might already exist, which is fine
            pass
    
    def _calculate_content_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file content only (filename-independent)
        This ensures the same content gets the same hash regardless of filename
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):  # Larger chunks for better performance
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"⚠️  Error calculating content hash for {file_path}: {e}")
            return None
    
    def _get_content_metadata(self, file_path: str) -> Dict:
        """Get comprehensive file metadata for content tracking"""
        path = Path(file_path)
        stat = path.stat()
        content_hash = self._calculate_content_hash(file_path)
        
        if not content_hash:
            return None
            
        return {
            "content_hash": content_hash,
            "current_file_path": str(path.absolute()),
            "current_file_name": path.name,
            "file_size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "file_extension": path.suffix.lower(),
            "last_checked": datetime.now()
        }
    
    def is_content_already_processed(self, file_path: str) -> bool:
        """
        Check if file content has already been processed based on content hash
        This is filename-independent - same content = already processed
        """
        try:
            content_hash = self._calculate_content_hash(file_path)
            if not content_hash:
                return False
            
            # Check if this content hash exists in our persistent collection
            existing_record = self.content_hash_collection.find_one({
                "content_hash": content_hash,
                "status": "processed"
            })
            
            if existing_record:
                # Update the current file path in case file was renamed/moved
                self.content_hash_collection.update_one(
                    {"content_hash": content_hash},
                    {
                        "$set": {
                            "current_file_path": str(Path(file_path).absolute()),
                            "current_file_name": Path(file_path).name,
                            "last_seen": datetime.now()
                        }
                    }
                )
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Error checking content status: {e}")
            return False
    
    def mark_content_as_processed(self, file_path: str, document_count: int = 0, chunk_count: int = 0):
        """
        Mark file content as processed in the persistent hash collection
        This creates a permanent record that survives collection clears
        """
        try:
            metadata = self._get_content_metadata(file_path)
            if not metadata:
                print(f"⚠️  Could not generate metadata for {file_path}")
                return False
            
            # Add processing information
            metadata.update({
                "processed_time": datetime.now(),
                "document_count": document_count,
                "chunk_count": chunk_count,
                "status": "processed",
                "processing_version": "2.0"  # Version for future compatibility
            })
            
            # Upsert the record based on content hash
            result = self.content_hash_collection.replace_one(
                {"content_hash": metadata["content_hash"]},
                metadata,
                upsert=True
            )
            
            action = "Updated" if result.matched_count > 0 else "Created"
            print(f"✅ {action} content hash record: {Path(file_path).name}")
            return True
            
        except Exception as e:
            print(f"⚠️  Error marking content as processed: {e}")
            return False
    
    def get_content_processing_stats(self) -> Dict:
        """Get statistics about processed content"""
        try:
            total_processed = self.content_hash_collection.count_documents({"status": "processed"})
            
            # Get file type distribution
            pipeline = [
                {"$match": {"status": "processed"}},
                {"$group": {
                    "_id": "$file_extension",
                    "count": {"$sum": 1},
                    "total_documents": {"$sum": "$document_count"},
                    "total_chunks": {"$sum": "$chunk_count"}
                }},
                {"$sort": {"count": -1}}
            ]
            
            file_types = list(self.content_hash_collection.aggregate(pipeline))
            
            return {
                "total_processed_files": total_processed,
                "file_type_distribution": file_types,
                "collection_name": "content_hashes"
            }
            
        except Exception as e:
            print(f"⚠️  Error getting processing stats: {e}")
            return {"error": str(e)}
    
    def find_duplicate_content(self) -> List[Dict]:
        """Find files with identical content (same hash, different paths)"""
        try:
            pipeline = [
                {"$match": {"status": "processed"}},
                {"$group": {
                    "_id": "$content_hash",
                    "files": {
                        "$push": {
                            "file_path": "$current_file_path",
                            "file_name": "$current_file_name",
                            "processed_time": "$processed_time"
                        }
                    },
                    "count": {"$sum": 1}
                }},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            duplicates = list(self.content_hash_collection.aggregate(pipeline))
            return duplicates
            
        except Exception as e:
            print(f"⚠️  Error finding duplicates: {e}")
            return []
    
    def cleanup_orphaned_hashes(self) -> int:
        """Remove hash records for files that no longer exist"""
        try:
            removed_count = 0
            
            # Get all processed records
            records = self.content_hash_collection.find({"status": "processed"})
            
            for record in records:
                file_path = record.get("current_file_path")
                if file_path and not Path(file_path).exists():
                    # File no longer exists, mark as orphaned or remove
                    self.content_hash_collection.update_one(
                        {"_id": record["_id"]},
                        {"$set": {"status": "orphaned", "orphaned_time": datetime.now()}}
                    )
                    removed_count += 1
            
            print(f"🧹 Marked {removed_count} orphaned hash records")
            return removed_count
            
        except Exception as e:
            print(f"⚠️  Error cleaning up orphaned hashes: {e}")
            return 0
    
    
    def load_queries_from_file(self, queries_file_path: str) -> List[str]:
        """Load queries from a text file (one query per line)"""
        try:
            queries = []
            with open(queries_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        queries.append(line)
            
            print(f"📝 Loaded {len(queries)} queries from {queries_file_path}")
            return queries
            
        except Exception as e:
            print(f"❌ Error loading queries from file: {e}")
            return []
    
    def process_queries_from_file(self, queries_file_path: str, output_file_path: str = None):
        """Process all queries from a file and optionally save results"""
        queries = self.load_queries_from_file(queries_file_path)
        
        if not queries:
            print("No queries to process")
            return
        
        results = []
        
        print(f"\n🔍 Processing {len(queries)} queries...")
        print("=" * 60)
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            print("-" * 50)
            
            try:
                result = self.query(query)
                answer = result['answer']
                source_count = len(result['source_documents'])
                
                print(f"📝 Answer: {answer}")
                print(f"📚 Sources: {source_count} documents")
                
                # Store result
                query_result = {
                    "query": query,
                    "answer": answer,
                    "source_count": source_count,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(query_result)
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save results to file if specified
        if output_file_path:
            try:
                import json
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Results saved to: {output_file_path}")
            except Exception as e:
                print(f"⚠️  Error saving results: {e}")
        
        return results

    def close_connection(self):
        """Close the MongoDB connection"""
        self.client.close()
