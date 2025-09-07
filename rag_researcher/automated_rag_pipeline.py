"""
Automated RAG Pipeline

This script continuously monitors an input folder for new documents,
processes them through the RAG pipeline, and maintains a persistent vector database.
After processing, it cleans up the input folder and can retrieve directly from
the existing vector database when no new files are present.
"""

import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_loader import DataLoader
from modules.text_cleaner import TextCleaner
from modules.chunker import TextChunker
from modules.embedder import TextEmbedder
from modules.vector_store import VectorStore
from modules.retriever import Retriever

class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events in the input directory."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.last_processed = time.time()
        
    def on_created(self, event):
        if not event.is_directory:
            # Wait a bit to ensure file is fully written
            time.sleep(2)
            self.pipeline.logger.info(f"New file detected: {event.src_path}")
            self.pipeline.run_once()
    
    def on_modified(self, event):
        if not event.is_directory and time.time() - self.last_processed > 5:
            # Avoid processing too frequently
            time.sleep(2)
            self.pipeline.logger.info(f"File modified: {event.src_path}")
            self.pipeline.run_once()
            self.last_processed = time.time()

class AutomatedRAGPipeline:
    """Automated RAG Pipeline with folder monitoring and persistent vector database."""
    
    def __init__(self, input_dir: str, output_dir: str, vector_db_path: str, 
                 embedding_provider: str = "nomic", embedding_model: str = "nomic-embed-text:latest",
                 chunk_size: int = 500, chunk_overlap: int = 50, monitor_interval: int = 30):
        """
        Initialize the automated RAG pipeline.
        
        Args:
            input_dir: Directory to monitor for input files
            output_dir: Directory for output files
            vector_db_path: Path to persistent vector database
            embedding_provider: Embedding provider (ollama, sentence_transformers, openai)
            embedding_model: Embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            monitor_interval: Seconds between folder checks
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.vector_db_path = Path(vector_db_path)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.monitor_interval = monitor_interval
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.text_cleaner = TextCleaner()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap_size=chunk_overlap
        )
        # Initialize embedder with proper Nomic support
        if embedding_provider == "ollama" or embedding_provider == "nomic":
            self.embedder = TextEmbedder(
                model_name=embedding_model,
                model_provider="nomic",
                cache_dir=str(self.output_dir / "embeddings_cache")
            )
        else:
            self.embedder = TextEmbedder(
                model_name=embedding_model,
                model_provider=embedding_provider,
                cache_dir=str(self.output_dir / "embeddings_cache")
            )
        
        # Initialize or load vector store
        self.vector_store = self._initialize_vector_store()
        self.retriever = Retriever(self.vector_store, self.embedder)
        
        # Log embedding configuration
        if embedding_provider == "ollama" or embedding_provider == "nomic":
            self.logger.info(f"Using Nomic embedding model: {embedding_model}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "automated_pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_vector_store(self) -> VectorStore:
        """Initialize or load existing vector store."""
        if self.vector_db_path.exists():
            try:
                self.logger.info(f"Loading existing vector database from {self.vector_db_path}")
                vector_store = VectorStore.load(str(self.vector_db_path))
                doc_count = len(vector_store.documents) if hasattr(vector_store, 'documents') else 0
                self.logger.info(f"Loaded vector store with {doc_count} documents")
                return vector_store
            except Exception as e:
                self.logger.error(f"Failed to load existing vector store: {e}")
                self.logger.info("Creating new vector database")
                return VectorStore()
        else:
            self.logger.info(f"Creating new vector database at {self.vector_db_path}")
            return VectorStore()
            
    def _get_input_files(self) -> List[Path]:
        """Get list of files in input directory."""
        supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
        files = []
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
                
        return files
        
    def _process_documents(self, file_paths: List[Path]) -> bool:
        """Process documents through the RAG pipeline."""
        try:
            self.logger.info(f"Processing {len(file_paths)} documents...")
            
            # Load documents
            documents = []
            for file_path in file_paths:
                self.logger.info(f"Loading document: {file_path.name}")
                doc = self.data_loader.load_file(file_path)
                documents.append(doc)
                
            self.logger.info(f"Loaded {len(documents)} documents")
            
            # Clean documents
            self.logger.info("Cleaning documents...")
            cleaned_documents = self.text_cleaner.clean_documents(documents)
            
            # Save cleaned documents
            for doc in cleaned_documents:
                cleaned_file = self.output_dir / f"{Path(doc['filename']).stem}_cleaned.txt"
                with open(cleaned_file, 'w', encoding='utf-8') as f:
                    f.write(doc['content'])
                    
            # Chunk documents
            self.logger.info("Chunking documents...")
            all_chunks = []
            for doc in cleaned_documents:
                chunked_doc = self.chunker.chunk_document(doc)
                
                # Extract individual chunks for embedding
                for chunk_obj in chunked_doc['chunks']:
                    # Create chunk dict for embedding with required fields
                    chunk_dict = {
                        'text': chunk_obj['content'],
                        'filename': chunked_doc['filename'],
                        'chunk_id': chunk_obj['chunk_id'],
                        'size_chars': chunk_obj['size_chars']
                    }
                    all_chunks.append(chunk_dict)
                
                # Save chunks
                chunks_dir = self.output_dir / f"{Path(doc['filename']).stem}_chunks"
                chunks_dir.mkdir(exist_ok=True)
                
                for i, chunk_obj in enumerate(chunked_doc['chunks'], 1):
                    chunk_file = chunks_dir / f"chunk_{i:04d}_{Path(doc['filename']).stem}.txt"
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        f.write(chunk_obj['content'])
                        
                # Save metadata
                metadata_file = chunks_dir / "_metadata.txt"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    f.write(f"Total chunks: {chunked_doc['total_chunks']}\n")
                    f.write(f"Chunk size: {self.chunk_size}\n")
                    f.write(f"Chunk overlap: {self.chunk_overlap}\n")
                    f.write(f"Source file: {doc['filename']}\n")
                    
            self.logger.info(f"Created {len(all_chunks)} chunks")
            
            # Generate embeddings
            self.logger.info("Generating embeddings...")
            embedded_chunks = self.embedder.embed_documents(all_chunks)
            self.logger.info(f"Generated {len(embedded_chunks)} embeddings")
            
            # Add to vector store
            self.logger.info("Adding embeddings to vector store...")
            self.vector_store.add_documents(embedded_chunks)
            
            # Save vector store
            self.logger.info(f"Saving vector database to {self.vector_db_path}")
            self.vector_store.save(str(self.vector_db_path))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            return False
            
    def _cleanup_input_folder(self, processed_files: List[Path]):
        """Remove processed files from input folder."""
        try:
            self.logger.info("Cleaning up input folder...")
            for file_path in processed_files:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.info(f"Removed: {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error cleaning up input folder: {str(e)}")
            
    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> dict:
        """Retrieve relevant documents for a query."""
        try:
            self.logger.info(f"Retrieving documents for query: '{query}'")
            
            # Perform retrieval
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                reranking_enabled=True,
                hybrid_search_enabled=True
            )
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
            results_dir = self.output_dir / "retriever" / f"{timestamp}_{query_safe.replace(' ', '_')}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save query
            with open(results_dir / "query.txt", 'w', encoding='utf-8') as f:
                f.write(query)
                
            # Save results
            import json
            import numpy as np
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            results_serializable = convert_numpy(results)
            results_data = {
                "query": query,
                "results": results_serializable,
                "total_results": len(results)
            }
            with open(results_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
                
            # Create formatted results file
            with open(results_dir / "results.txt", 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"Result {i}:\n")
                    f.write(f"Score: {result.get('similarity_score', 'N/A')}\n")
                    content = result.get('text', result.get('content', 'N/A'))
                    f.write(f"Content: {content[:500]}{'...' if len(str(content)) > 500 else ''}\n")
                    f.write(f"Filename: {result.get('filename', 'N/A')}\n")
                    f.write(f"Chunk ID: {result.get('chunk_id', 'N/A')}\n")
                    f.write("-" * 50 + "\n\n")
            
            self.logger.info(f"Retrieved {len(results)} documents")
            self.logger.info(f"Results saved to: {results_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            return {"results": [], "error": str(e)}
            
    def run_once(self) -> bool:
        """Run the pipeline once - check for files and process if found."""
        input_files = self._get_input_files()
        
        if input_files:
            self.logger.info(f"Found {len(input_files)} files to process")
            success = self._process_documents(input_files)
            
            if success:
                self._cleanup_input_folder(input_files)
                self.logger.info("Processing completed successfully")
                return True
            else:
                self.logger.error("Processing failed")
                return False
        else:
            self.logger.info("No new files found in input folder")
            return True
            
    def run_continuous(self):
        """Run the pipeline continuously, monitoring the input folder with file system events."""
        self.logger.info(f"Starting continuous monitoring of {self.input_dir}")
        
        # Set up file system watcher
        event_handler = FileChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.input_dir), recursive=False)
        
        try:
            observer.start()
            self.logger.info("File system monitoring started")
            
            # Also run periodic checks as backup
            while True:
                time.sleep(self.monitor_interval)
                if self._get_input_files():
                    self.run_once()
                    
        except KeyboardInterrupt:
            self.logger.info("Pipeline stopped by user")
            observer.stop()
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            observer.stop()
        finally:
            observer.join()
            
    def status(self) -> dict:
        """Get pipeline status information."""
        input_files = self._get_input_files()
        vector_db_exists = self.vector_db_path.exists()
        
        if vector_db_exists:
            try:
                vector_count = len(self.vector_store.embeddings) if hasattr(self.vector_store, 'embeddings') else 0
            except:
                vector_count = "Unknown"
        else:
            vector_count = 0
            
        return {
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "vector_database": str(self.vector_db_path),
            "vector_db_exists": vector_db_exists,
            "vector_count": vector_count,
            "pending_files": len(input_files),
            "pending_file_names": [f.name for f in input_files],
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "monitor_interval": self.monitor_interval
        }

def main():
    parser = argparse.ArgumentParser(description="Automated RAG Pipeline")
    parser.add_argument("--input", default="input", help="Input directory to monitor")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--vector-db", default="vector_db/rag_database.pkl", help="Vector database path")
    parser.add_argument("--embedding-provider", default="ollama", 
                       choices=["ollama", "sentence_transformers", "openai"],
                       help="Embedding provider")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Embedding model")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--monitor-interval", type=int, default=30, help="Monitor interval in seconds")
    parser.add_argument("--mode", choices=["once", "continuous", "retrieve", "status"], 
                       default="once", help="Operation mode")
    parser.add_argument("--query", help="Query for retrieval mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutomatedRAGPipeline(
        input_dir=args.input,
        output_dir=args.output,
        vector_db_path=args.vector_db,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        monitor_interval=args.monitor_interval
    )
    
    if args.mode == "once":
        success = pipeline.run_once()
        sys.exit(0 if success else 1)
        
    elif args.mode == "continuous":
        pipeline.run_continuous()
        
    elif args.mode == "retrieve":
        if not args.query:
            print("Error: --query is required for retrieve mode")
            sys.exit(1)
            
        results = pipeline.retrieve(args.query, top_k=args.top_k)
        if isinstance(results, list):
            print(f"Retrieved {len(results)} results for query: '{args.query}'")
            if results:
                print("\nTop result:")
                print(f"Score: {results[0].get('similarity_score', 'N/A')}")
                content = results[0].get('text', results[0].get('content', 'N/A'))
                print(f"Content: {str(content)[:200]}...")
                print(f"Filename: {results[0].get('filename', 'N/A')}")
        else:
            print(f"Retrieval failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    elif args.mode == "status":
        status = pipeline.status()
        print("\n=== Pipeline Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        print()

if __name__ == "__main__":
    main()