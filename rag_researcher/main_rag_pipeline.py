#!/usr/bin/env python3
"""
Simple RAG Pipeline Script

This script implements a streamlined RAG (Retrieval-Augmented Generation) pipeline that processes
documents through a series of stages to prepare them for use in retrieval systems. The pipeline
follows these key steps:

1. Data Loading: Reads files from the input directory, supporting various document formats
2. Text Cleaning: Preprocesses and normalizes text data to improve quality
3. Chunking: Splits cleaned text into manageable chunks suitable for embedding and retrieval
4. Output: Saves both cleaned and chunked text files to the output directory

The pipeline is designed to be simple yet effective, focusing on the core preprocessing steps
needed for RAG systems without unnecessary complexity. It handles the critical task of
transforming raw documents into clean, properly sized text chunks that can be effectively
embedded and retrieved.

Usage:
    python main_rag_pipeline.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--verbose]

Author: RAG System
"""

import os
import sys
import json      # For saving chunk data in JSON format
import time      # For timestamps in metadata
import argparse  # For command-line argument parsing
import logging   # For structured logging throughout the pipeline
import time      # For performance timing measurements
from pathlib import Path  # For cross-platform path handling
from typing import Dict, List, Optional, Any  # Type hints for better code documentation

# Add modules directory to path to enable imports from the modules package
# This allows the script to find our custom modules regardless of how it's executed
sys.path.append(str(Path(__file__).parent / "modules"))

# Import our custom modules that implement the core pipeline components
from data_loader import DataLoader    # Handles loading documents from various sources
from text_cleaner import TextCleaner  # Handles text preprocessing and normalization
from chunker import TextChunker       # Handles splitting text into appropriate chunks
from embedder import TextEmbedder     # Handles converting text chunks to vector embeddings
from vector_store import VectorStore  # Handles storing and retrieving vector embeddings
from retriever import Retriever       # Handles retrieving relevant documents based on queries

# Configure logging with a standardized format for consistent output across all pipeline components
# This setup ensures that all log messages include timestamp, logger name, level, and message
logging.basicConfig(
    level=logging.INFO,  # Set default logging level to INFO for normal operation
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Standard format with timestamp
    handlers=[
        logging.StreamHandler()  # Output logs to console/stdout for immediate visibility
    ]
)
# Create a logger specific to this module
logger = logging.getLogger(__name__)

class RAGPipelineConfig:
    """
    Configuration class for the RAG Pipeline.
    
    This class centralizes all configuration parameters for the RAG pipeline,
    making it easier to customize the pipeline behavior without modifying the
    core implementation. It includes settings for input/output directories,
    document processing, chunking, embedding, and vector storage.
    """
    
    def __init__(self,
                 input_dir: str = "input",
                 output_dir: str = "output",
                 # Document loading settings
                 supported_formats: List[str] = None,
                 # Document processing settings
                 ocr_enabled: bool = False,
                 ocr_language: str = "  eng",
                 # Chunking settings
                 chunk_size: int = 600,
                 chunk_overlap: int = 150,
                 # Embedding settings
                 embedding_model: str = "nomic-embed-text",
                 embedding_provider: str = "nomic",
                 embedding_dim: int = 768,
                 embedding_api_key: Optional[str] = None,
                 # Vector database settings
                 distance_metric: str = "cosine",
                 # Retriever settings
                 retriever_top_k: int = 5,
                 retriever_similarity_threshold: float = 0.0,
                 reranking_enabled: bool = True,
                 hybrid_search_enabled: bool = True,
                 hybrid_search_weight: float = 0.7):
        """
        Initialize the RAG Pipeline configuration with customizable parameters.
        
        Args:
            input_dir (str): Directory containing input documents to be processed
            output_dir (str): Directory where processed outputs will be saved
            supported_formats (List[str]): List of supported file formats (default: ["txt", "md", "pdf"])
            ocr_enabled (bool): Whether to use OCR for image-based PDFs
            ocr_language (str): Language for OCR processing
            chunk_size (int): Target size in characters for each chunk
            chunk_overlap (int): Overlap between chunks to maintain context
            embedding_model (str): Name of the embedding model to use
            embedding_provider (str): Provider for embeddings (sentence_transformers, nomic, openai)
            embedding_dim (int): Dimension of the embedding vectors
            embedding_api_key (str): API key for commercial embedding providers
            distance_metric (str): Similarity metric for vector comparison (cosine, euclidean, dot)
            retriever_top_k (int): Number of results to return in retrieval (default: 5)
            retriever_similarity_threshold (float): Minimum similarity score for retrieval results (default: 0.0)
        """
        # Input/output settings
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Document loading settings
        self.supported_formats = supported_formats or ["txt", "md", "pdf"]
        
        # Document processing settings
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        
        # Chunking settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Embedding settings
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_dim = embedding_dim
        self.embedding_api_key = embedding_api_key
        
        # Vector database settings
        self.distance_metric = distance_metric
        
        # Retriever settings
        self.retriever_top_k = retriever_top_k
        self.retriever_similarity_threshold = retriever_similarity_threshold
        self.reranking_enabled = reranking_enabled
        self.hybrid_search_enabled = hybrid_search_enabled
        self.hybrid_search_weight = hybrid_search_weight


class RAGPipeline:
    """
    Simple RAG Pipeline class that orchestrates the complete document processing workflow.
    
    This class serves as the main coordinator for the entire RAG pipeline, connecting all the
    individual components (data loading, text cleaning, and chunking) into a cohesive workflow.
    It manages the flow of data between components and handles the overall process from raw
    document ingestion to producing cleaned and properly chunked text suitable for embedding
    and retrieval systems.
    
    The pipeline follows a sequential processing approach:
    1. Load documents from the input directory using DataLoader
    2. Clean and normalize the text using TextCleaner
    3. Split the cleaned text into appropriate chunks using TextChunker
    4. Embed chunks and store them in the vector database
    5. Save both the cleaned and chunked outputs to the output directory
    
    This design allows for easy extension or modification of individual components while
    maintaining the overall workflow structure.
    """
    
    def __init__(self, 
                 input_dir: str = "input",
                 output_dir: str = "output",
                 config: Optional[RAGPipelineConfig] = None):
        """
        Initialize the RAG Pipeline with input and output directory configurations.
        
        This constructor sets up the pipeline environment, creates necessary directories,
        and initializes all the required processing components with their default settings.
        The default configuration is suitable for most general document processing tasks,
        but can be customized by providing a RAGPipelineConfig object.
        
        Args:
            input_dir (str): Directory containing input documents to be processed
            output_dir (str): Directory where processed outputs will be saved
            config (RAGPipelineConfig): Configuration object for the pipeline
        """
        # Use provided config or create a new one with default settings
        self.config = config or RAGPipelineConfig(input_dir=input_dir, output_dir=output_dir)
        
        # Convert string paths to Path objects for better cross-platform compatibility
        self.input_dir = Path(self.config.input_dir)   # Source directory for documents
        self.output_dir = Path(self.config.output_dir) # Target directory for processed outputs
        
        # Create output directory if it doesn't exist
        # parents=True ensures all parent directories are created as needed
        # exist_ok=True prevents errors if the directory already exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all pipeline components with settings from config
        # This sets up the data loader, text cleaner, chunker, embedder, and vector store
        self._initialize_components()
        
        # Log initialization information for debugging and monitoring
        logger.info(f"RAG Pipeline initialized:")
        logger.info(f"  - Input directory: {self.input_dir}")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info(f"  - Embedding model: {self.config.embedding_model} ({self.config.embedding_provider})")
        logger.info(f"  - Vector similarity: {self.config.distance_metric}")
    
    def _initialize_components(self):
        """
        Initialize pipeline components based on configuration.
        
        This method creates instances of all core pipeline components using the
        configuration parameters from the RAGPipelineConfig object. Each component
        is configured according to the settings provided, allowing for customization
        of the pipeline behavior without modifying the core implementation.
        
        The method initializes:
        - DataLoader: Responsible for loading documents from the input directory
        - TextCleaner: Handles text preprocessing and normalization tasks
        - TextChunker: Manages splitting documents into appropriately sized chunks
        - TextEmbedder: Converts text chunks into vector embeddings
        - VectorStore: Stores and retrieves vector embeddings for similarity search
        
        If a vector store file exists in the output directory, it will be loaded instead
        of creating a new empty vector store.
        """
        # Initialize Data Loader with the input directory path
        # This component will handle finding and loading documents from various formats
        self.data_loader = DataLoader(
            input_dir=str(self.input_dir)
        )
        
        # Initialize Text Cleaner with settings optimized for maintaining text quality
        # while removing problematic elements that could affect downstream processing
        self.text_cleaner = TextCleaner(
            remove_extra_whitespace=True,    # Remove redundant spaces for cleaner text
            normalize_unicode=True,          # Standardize unicode characters for consistency
            remove_special_chars=False,      # Preserve special characters that may be meaningful
            preserve_line_breaks=True        # Maintain document structure with line breaks
        )
        
        # Initialize Text Chunker with settings from configuration
        # This creates appropriately sized chunks that preserve context and meaning
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,           # Target size from config
            overlap_size=self.config.chunk_overlap,      # Overlap from config
            chunking_method='paragraph',                 # Use paragraph boundaries as natural chunk divisions
            preserve_sentences=True,                     # Never break sentences across chunk boundaries
            min_chunk_size=self.config.chunk_size // 3   # Minimum chunk size to avoid tiny fragments
        )
        
        # Initialize Text Embedder with settings from configuration
        # This component transforms text chunks into numerical vectors for similarity search
        self.text_embedder = TextEmbedder(
            model_name=self.config.embedding_model,       # Embedding model from config
            model_provider=self.config.embedding_provider, # Provider from config
            embedding_dim=self.config.embedding_dim,      # Dimension from config
            api_key=self.config.embedding_api_key,        # API key from config
            batch_size=32,                                # Process chunks in batches for efficiency
            cache_dir=self.output_dir / "embeddings_cache" # Cache embeddings to avoid redundant computation
        )
        
        # Check if a vector store file exists and load it if available
        vector_store_path = self.output_dir / "vector_store.pkl"
        if vector_store_path.exists():
            try:
                logger.info(f"Loading existing vector store from {vector_store_path}")
                self.vector_store = VectorStore.load(vector_store_path)
                logger.info(f"Loaded vector store with {len(self.vector_store.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                logger.info("Creating new vector store")
                self.vector_store = VectorStore(
                    distance_metric=self.config.distance_metric
                )
        else:
            # Initialize Vector Store with similarity metric from configuration
            # This component enables similarity search for retrieval
            logger.info("No existing vector store found. Creating new one.")
            self.vector_store = VectorStore(
                distance_metric=self.config.distance_metric    # Similarity metric from config
            )
        
        # Initialize Enhanced Retriever with vector store and embedder
        # This component handles retrieving semantically relevant documents based on queries
        self.retriever = Retriever(
            vector_store=self.vector_store,               # Vector store for document retrieval
            embedder=self.text_embedder,                 # Embedder for query embedding
            top_k=self.config.retriever_top_k,           # Number of results to return
            similarity_threshold=self.config.retriever_similarity_threshold,  # Minimum similarity score
            reranking_enabled=self.config.reranking_enabled,  # Enable reranking for better results
            hybrid_search_enabled=self.config.hybrid_search_enabled,  # Enable hybrid search (semantic + keyword)
            hybrid_search_weight=self.config.hybrid_search_weight  # Weight for balancing semantic vs keyword search
        )
        
        logger.info("Pipeline components initialized successfully with enhanced semantic retriever")
    
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load documents from the input directory using the DataLoader component.
        
        This method represents the first step in the RAG pipeline. It scans the input
        directory for supported file types (TXT, MD, PDF), loads their content, and
        creates structured document objects with metadata. The method provides detailed
        logging about the files found and loaded to help with monitoring and debugging.
        
        Returns:
            List[Dict[str, str]]: A list of document dictionaries, each containing:
                - 'filename': Original filename
                - 'filepath': Full path to the source file
                - 'content': Raw text content of the document
                - 'size_chars': Character count of the content
                - 'file_type': Original file format (txt, md, pdf)
        """
        # Log the start of the document loading phase
        logger.info("=== STEP 1: LOADING DOCUMENTS ===")
        
        # Get statistics about available files in the input directory
        # This helps understand what types of documents are being processed
        stats = self.data_loader.get_stats()
        logger.info(f"Found {stats['total_files']} supported files:")
        logger.info(f"  - Text files (.txt): {stats['txt_files']}")
        logger.info(f"  - Markdown files (.md): {stats['md_files']}")
        logger.info(f"  - PDF files (.pdf): {stats['pdf_files']}")
        
        # Load all documents from the input directory
        # The DataLoader handles different file formats appropriately
        documents = self.data_loader.load_all_files()
        
        # Handle the case where no documents were successfully loaded
        if not documents:
            logger.warning("No documents were loaded successfully")
            return []
        
        # Calculate and log statistics about the loaded documents
        # This provides insight into the volume and size of data being processed
        total_chars = sum(doc['size_chars'] for doc in documents)
        logger.info(f"Successfully loaded {len(documents)} documents:")
        logger.info(f"  - Total characters: {total_chars:,}")
        logger.info(f"  - Average document size: {total_chars // len(documents):,} chars")
        
        return documents
    
    def clean_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Clean and preprocess documents using the TextCleaner component.
        
        This method represents the second step in the RAG pipeline. It takes the raw
        documents loaded in the previous step and applies various text cleaning and
        normalization operations to improve text quality. The cleaning process helps
        remove noise and standardize the text format, which is crucial for effective
        chunking and retrieval in later stages.
        
        The cleaning operations include:
        - Removing excessive whitespace
        - Normalizing Unicode characters
        - Preserving meaningful line breaks
        - Optionally removing special characters
        
        Args:
            documents (List[Dict[str, str]]): Raw documents from the loading step
            
        Returns:
            List[Dict[str, str]]: Cleaned documents with the same structure as input
                but with normalized and cleaned content
        """
        # Log the start of the document cleaning phase
        logger.info("=== STEP 2: CLEANING DOCUMENTS ===")
        
        # Handle the case where no documents were provided
        if not documents:
            logger.warning("No documents to clean")
            return []
        
        # Process all documents through the TextCleaner component
        # This applies all configured cleaning operations to each document
        cleaned_documents = self.text_cleaner.clean_documents(documents)
        
        # Calculate statistics to measure the impact of cleaning
        # This helps understand how much the text was modified during cleaning
        total_original_chars = sum(len(doc['content']) for doc in documents)
        total_cleaned_chars = sum(len(doc['content']) for doc in cleaned_documents)
        reduction_percentage = ((total_original_chars - total_cleaned_chars) / total_original_chars * 100) if total_original_chars > 0 else 0
        
        logger.info(f"Text cleaning completed:")
        logger.info(f"  - Documents processed: {len(cleaned_documents)}")
        logger.info(f"  - Size reduction: {reduction_percentage:.1f}%")
        
        return cleaned_documents
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict]:
        """
        Chunk cleaned documents into smaller pieces using the TextChunker component.
        
        This method represents the third and most critical step in the RAG pipeline.
        It takes the cleaned documents from the previous step and splits them into
        smaller, semantically meaningful chunks that are suitable for embedding and
        retrieval. The chunking process is essential for effective RAG systems as it
        determines the granularity of information that can be retrieved.
        
        The chunking process uses the configured strategy (paragraph-based by default)
        and ensures that chunks are properly sized with appropriate overlap. It also
        preserves sentence boundaries to maintain semantic coherence within chunks.
        
        Args:
            documents (List[Dict[str, str]]): Cleaned documents from the cleaning step
            
        Returns:
            List[Dict]: Chunked documents with extensive metadata, including:
                - Document-level metadata (filename, filepath, etc.)
                - Chunking statistics (total chunks, chunk sizes, etc.)
                - Individual chunks with their content and position information
        """
        # Log the start of the document chunking phase
        logger.info("=== STEP 3: CHUNKING DOCUMENTS ===")
        
        # Handle the case where no documents were provided
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        # Process all documents through the TextChunker component
        # This splits each document into appropriately sized chunks based on configuration
        chunked_documents = self.text_chunker.chunk_documents(documents)
        
        # Calculate and log statistics about the chunking results
        # This helps understand the volume and distribution of chunks created
        total_chunks = sum(doc.get('total_chunks', 0) for doc in chunked_documents)
        logger.info(f"Text chunking completed:")
        logger.info(f"  - Documents processed: {len(chunked_documents)}")
        logger.info(f"  - Total chunks created: {total_chunks}")
        
        return chunked_documents
        
    def embed_chunks(self, chunked_documents: List[Dict]) -> VectorStore:
        """
        Convert text chunks to vector embeddings and store them in the vector database.
        
        This method processes all chunks from the provided documents, converts them to
        vector embeddings using the TextEmbedder component, and stores them in the
        VectorStore for later retrieval.
        
        Args:
            chunked_documents (List[Dict]): Chunked documents from the chunking step
            
        Returns:
            VectorStore: The populated vector store containing all embedded chunks
        """
        logger.info("=== STEP 4: EMBEDDING CHUNKS ===")
        
        # Handle the case where no documents were provided
        if not chunked_documents:
            logger.warning("No chunked documents to embed")
            return self.vector_store
        
        # Prepare a list to collect all chunks for batch processing
        all_chunks = []
        
        # Collect all chunks from all documents with their metadata
        for doc in chunked_documents:
            for chunk in doc['chunks']:
                # Create metadata combining document and chunk information
                metadata = {
                    'filename': doc['filename'],
                    'filepath': doc.get('filepath', ''),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'start_char_idx': chunk.get('start_char_idx', 0),
                    'end_char_idx': chunk.get('end_char_idx', 0),
                    'page': chunk.get('page', None),
                    'section': chunk.get('section', None)
                }
                
                # Add the chunk to our collection with its metadata
                all_chunks.append({
                    'text': chunk['content'],
                    'metadata': metadata
                })
        
        logger.info(f"Processing {len(all_chunks)} chunks for embedding")
        
        # Generate embeddings for all chunks
        embeddings = self.text_embedder.embed_documents(all_chunks)
        
        # Store the embeddings in the vector store
        self.vector_store.add_documents(embeddings)
        
        logger.info(f"Successfully embedded and stored {len(embeddings)} chunks")
        
        # Save the vector store to disk for persistence
        vector_store_path = self.output_dir / "vector_store.pkl"
        self.vector_store.save(str(vector_store_path))
        logger.info(f"Vector store saved to {vector_store_path}")
        
        return self.vector_store
    
    def retrieve(self, query: str, top_k: Optional[int] = None, similarity_threshold: Optional[float] = None, 
                save_results: bool = True, reranking_enabled: Optional[bool] = None, 
                hybrid_search_enabled: Optional[bool] = None, hybrid_search_weight: Optional[float] = None):
        """
        Retrieve semantically relevant document chunks based on a query.
        
        This method uses the enhanced retriever component to find document chunks that are most
        semantically similar to the given query. It leverages advanced techniques like reranking 
        and hybrid search (combining semantic and keyword search) to improve result quality.
        Optionally saves results to the output directory.
        
        Args:
            query (str): The query text to search for
            top_k (Optional[int]): Number of results to return, overrides config default
            similarity_threshold (Optional[float]): Minimum similarity score, overrides config default
            save_results (bool): Whether to save results to output/retriever directory
            reranking_enabled (Optional[bool]): Whether to use reranking for improved results
            hybrid_search_enabled (Optional[bool]): Whether to use hybrid search (semantic + keyword)
            hybrid_search_weight (Optional[float]): Weight for balancing semantic vs keyword search (0-1)
            
        Returns:
            List[Dict[str, Any]]: The most relevant document chunks with similarity scores
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        # Initialize retriever with the vector store if not already initialized
        if not hasattr(self, 'retriever') or self.retriever is None:
            # Make sure we have a vector store
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                vector_store_path = self.output_dir / "vector_store.pkl"
                if vector_store_path.exists():
                    logger.info(f"Loading vector store from {vector_store_path}")
                    self.vector_store = VectorStore.load(vector_store_path)
                else:
                    logger.error("No vector store available for retrieval")
                    return []
            
            # Initialize the retriever with the vector store and enhanced settings
            self.retriever = Retriever(
                vector_store=self.vector_store,
                embedder=self.text_embedder,
                top_k=self.config.retriever_top_k,
                similarity_threshold=self.config.retriever_similarity_threshold,
                reranking_enabled=getattr(self.config, 'reranking_enabled', True),
                hybrid_search_enabled=getattr(self.config, 'hybrid_search_enabled', True),
                hybrid_search_weight=getattr(self.config, 'hybrid_search_weight', 0.7)
            )
            logger.info("Initialized enhanced retriever with vector store")
        
        # Use the retriever to find relevant documents with enhanced semantic search
        results = self.retriever.retrieve_and_format(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_metadata=True,
            reranking_enabled=reranking_enabled,
            hybrid_search_enabled=hybrid_search_enabled,
            hybrid_search_weight=hybrid_search_weight
        )
        
        logger.info(f"Retrieved {len(results)} semantically relevant documents")
        
        # Save results to output/retriever directory if requested
        if save_results and results:
            self._save_retriever_results(query, results)
            
        return results
        
    def _has_input_files(self) -> bool:
        """
        Check if there are any files in the input directory.
        
        Returns:
            bool: True if there are files in the input directory, False otherwise
        """
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return False
            
        # Check if there are any files in the input directory
        files = [f for f in input_dir.iterdir() if f.is_file()]
        if not files:
            logger.info(f"No files found in input directory: {input_dir}")
            return False
            
        logger.info(f"Found {len(files)} files in input directory: {input_dir}")
        return True
    
    def _cleanup_input_folder(self) -> None:
        """
        Clean up the input folder by removing all files after processing.
        """
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            return
            
        # Remove all files in the input directory
        files = [f for f in input_dir.iterdir() if f.is_file()]
        for file in files:
            try:
                file.unlink()
                logger.info(f"Removed file from input directory: {file}")
            except Exception as e:
                logger.error(f"Failed to remove file {file}: {e}")
                
        logger.info(f"Cleaned up input directory: {input_dir}")
    
    def _save_retriever_results(self, query: str, results: List[Dict[str, Any]]):
        """
        Save retriever results to the output/retriever directory.
        
        This method creates a directory for retriever results and saves the query results
        in both a summary file and individual result files.
        
        Args:
            query (str): The query that was used for retrieval
            results (List[Dict[str, Any]]): The retrieval results to save
        """
        # Create retriever directory if it doesn't exist
        retriever_dir = self.output_dir / "retriever"
        retriever_dir.mkdir(exist_ok=True)
        
        # Create a timestamp-based query directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Create a filename-friendly version of the query by removing invalid characters
        query_slug = query.lower().replace(" ", "_")
        # Remove characters that are not allowed in Windows filenames
        query_slug = "".join(c for c in query_slug if c.isalnum() or c in "_-")[:30]
        query_dir = retriever_dir / f"{timestamp}_{query_slug}"
        query_dir.mkdir(exist_ok=True)
        
        # Save the original query to a file
        query_file = query_dir / "query.txt"
        with open(query_file, 'w', encoding='utf-8') as f:
            f.write(query)
        
        # Save all results to a JSON file for easier programmatic access
        json_results = query_dir / "results.json"
        with open(json_results, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "config": {
                    "top_k": self.config.retriever_top_k,
                    "similarity_threshold": self.config.retriever_similarity_threshold,
                    "reranking_enabled": getattr(self.config, 'reranking_enabled', True),
                    "hybrid_search_enabled": getattr(self.config, 'hybrid_search_enabled', True),
                    "hybrid_search_weight": getattr(self.config, 'hybrid_search_weight', 0.7)
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        # Save query summary file
        summary_path = query_dir / "_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results: {len(results)}\n")
            f.write(f"Top-K: {self.config.retriever_top_k}\n")
            f.write(f"Similarity Threshold: {self.config.retriever_similarity_threshold}\n")
            f.write(f"Reranking Enabled: {getattr(self.config, 'reranking_enabled', True)}\n")
            f.write(f"Hybrid Search Enabled: {getattr(self.config, 'hybrid_search_enabled', True)}\n")
            f.write(f"Hybrid Search Weight: {getattr(self.config, 'hybrid_search_weight', 0.7)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write a summary of all results
            for i, result in enumerate(results):
                f.write(f"Result #{i+1} (Score: {result['similarity_score']:.4f})\n")
                if 'metadata' in result:
                    for key, value in result['metadata'].items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        # Save individual result files
        for i, result in enumerate(results):
            # Save as text file for human reading
            result_path = query_dir / f"result_{i+1:02d}.txt"
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"Result #{i+1} (Score: {result['similarity_score']:.4f})\n")
                f.write("=" * 80 + "\n")
                f.write(result['text'])
                f.write("\n" + "=" * 80 + "\n\n")
                
                # Write metadata if available
                if 'metadata' in result:
                    f.write("Metadata:\n")
                    for key, value in result['metadata'].items():
                        f.write(f"  {key}: {value}\n")
            
            # Save as JSON for programmatic access
            result_json_path = query_dir / f"result_{i+1:02d}.json"
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved retriever results to: {query_dir}")
    
    def save_outputs(self, chunked_documents: List[Dict]):
        """
        Save processed outputs to files in the output directory.
        
        This method represents the final step in the RAG pipeline. It takes the chunked
        documents from the previous step and saves both the cleaned text and the chunked
        text to separate files in the output directory. This allows for easy inspection
        and verification of the pipeline results, as well as providing the processed
        data in a format that can be used by downstream applications.
        
        For each document, two files are created:
        1. A cleaned text file containing the normalized text without chunking
        2. A chunked text file showing how the text was divided into chunks
        
        Args:
            chunked_documents (List[Dict]): Chunked documents from the chunking step
        """
        # Log the start of the output saving phase
        logger.info("=== STEP 4: SAVING OUTPUTS ===")
        
        # Handle the case where no documents were provided
        if not chunked_documents:
            logger.warning("No chunked documents to save")
            return
        
        # Process each document and save its outputs
        for doc in chunked_documents:
            # Extract the base filename without extension for creating output filenames
            # This preserves the original document name in the output files
            base_filename = os.path.basename(doc['filename'])
            base_name = os.path.splitext(base_filename)[0]
            
            # Reconstruct the cleaned text from the chunks
            # This is a workaround since we don't store the cleaned text separately
            # We concatenate all chunk contents with paragraph breaks between them
            original_content = ""
            for chunk in doc['chunks']:
                original_content += chunk['content'] + "\n\n"
            
            # Save the cleaned text to a file
            # This file contains the normalized text without chunking
            cleaned_text_path = self.output_dir / f"{base_name}_cleaned.txt"
            with open(cleaned_text_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            logger.info(f"Saved cleaned text to: {cleaned_text_path}")
            
            # Create individual files for each chunk with enhanced metadata
            # Each chunk is saved as a separate file with its metadata and content
            chunks_dir = self.output_dir / f"{base_name}_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            # Create a metadata summary file for all chunks
            metadata_path = chunks_dir / "_metadata.txt"
            with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                # Write document header information
                meta_file.write(f"Source Document: {doc['filename']}\n")
                meta_file.write(f"Total chunks: {doc.get('total_chunks', len(doc['chunks']))}\n")
                meta_file.write(f"Chunking method: {doc.get('chunking_method', self.text_chunker.chunking_method)}\n")
                meta_file.write(f"Chunk size: {doc.get('chunk_size', self.text_chunker.chunk_size)} chars\n")
                meta_file.write(f"Overlap size: {doc.get('overlap_size', self.text_chunker.overlap_size)} chars\n")
                meta_file.write(f"Processing timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                meta_file.write("=" * 50 + "\n\n")
                
                # Write a summary of all chunks
                meta_file.write("Chunk Summary:\n")
                for i, chunk in enumerate(doc['chunks']):
                    chunk_size = chunk.get('size_chars', len(chunk['content']))
                    meta_file.write(f"Chunk {i+1}: {chunk_size} chars")
                    
                    # Add position information if available
                    if 'start_char_idx' in chunk and 'end_char_idx' in chunk:
                        meta_file.write(f", position {chunk['start_char_idx']}-{chunk['end_char_idx']}")
                    
                    # Add page information if available
                    if 'page' in chunk:
                        meta_file.write(f", page {chunk['page']}")
                        
                    meta_file.write("\n")
            
            # Save each chunk as a separate file with metadata
            for i, chunk in enumerate(doc['chunks']):
                # Create a filename with chunk number and source document name
                chunk_filename = f"chunk_{i+1:04d}_{base_name}.txt"
                chunk_path = chunks_dir / chunk_filename
                
                with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
                    # Write chunk metadata header
                    chunk_file.write(f"SOURCE: {doc['filename']}\n")
                    chunk_file.write(f"CHUNK: {i+1} of {len(doc['chunks'])}\n")
                    chunk_file.write(f"SIZE: {chunk.get('size_chars', len(chunk['content']))} characters\n")
                    
                    # Include position information if available
                    if 'start_char_idx' in chunk and 'end_char_idx' in chunk:
                        chunk_file.write(f"POSITION: chars {chunk['start_char_idx']} to {chunk['end_char_idx']}\n")
                    
                    # Add any other available metadata
                    if 'section' in chunk:
                        chunk_file.write(f"SECTION: {chunk['section']}\n")
                    if 'page' in chunk:
                        chunk_file.write(f"PAGE: {chunk['page']}\n")
                    
                    # Add relationship information
                    if i > 0:
                        chunk_file.write(f"PREVIOUS: chunk_{i:04d}_{base_name}.txt\n")
                    if i < len(doc['chunks']) - 1:
                        chunk_file.write(f"NEXT: chunk_{i+2:04d}_{base_name}.txt\n")
                    
                    chunk_file.write("-" * 30 + "\n")
                    # Write the actual chunk content
                    chunk_file.write(chunk['content'])
            
            logger.info(f"Saved {len(doc['chunks'])} individual chunks with metadata to: {chunks_dir}")
        
        # Log completion of the saving process
        logger.info(f"All outputs saved to: {self.output_dir}")
    
    # Method _save_pipeline_stats removed as we're only saving cleaned and chunked text files
    
    def _has_input_files(self) -> bool:
        """
        Check if there are any files in the input directory.
        
        Returns:
            bool: True if there are files in the input directory, False otherwise
        """
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return False
            
        # Check if there are any files in the input directory
        files = [f for f in input_dir.iterdir() if f.is_file()]
        if not files:
            logger.info(f"No files found in input directory: {input_dir}")
            return False
            
        logger.info(f"Found {len(files)} files in input directory: {input_dir}")
        return True
    
    def _cleanup_input_folder(self) -> None:
        """
        Clean up the input folder by removing all files after processing.
        """
        input_dir = Path(self.config.input_dir)
        if not input_dir.exists():
            return
            
        # Remove all files in the input directory
        files = [f for f in input_dir.iterdir() if f.is_file()]
        for file in files:
            try:
                file.unlink()
                logger.info(f"Removed file from input directory: {file}")
            except Exception as e:
                logger.error(f"Failed to remove file {file}: {e}")
                
        logger.info(f"Cleaned up input directory: {input_dir}")
    
    def run(self, query: Optional[str] = None) -> bool:
        """
        Run the complete RAG pipeline from start to finish.
        
        This method orchestrates the entire pipeline process, executing each step in sequence
        and handling errors appropriately. It represents the main entry point for running the
        pipeline and ensures that each step is completed successfully before proceeding to
        the next one.
        
        The pipeline follows this sequence:
        1. Check if there are files in the input directory
        2. If files exist:
           a. Load documents from input directory
           b. Clean and preprocess text
           c. Split text into chunks
           d. Embed chunks and store them in the vector database
           e. Save cleaned text and chunked text files
           f. Clean up input folder
        3. If a query is provided, run retrieval using the vector store
        
        Args:
            query (Optional[str]): If provided, run retrieval with this query after processing
        
        The method includes error handling to gracefully abort the pipeline if any step fails,
        and provides detailed logging throughout the process for monitoring and debugging.
        
        Returns:
            bool: True if the pipeline completed successfully, False if any step failed
        """
        try:
            # Log the start of the pipeline execution
            logger.info("Starting RAG Pipeline...")
            # Record the start time for performance measurement
            start_time = time.time()
            
            # Check if vector store exists
            vector_store_path = self.output_dir / "vector_store.pkl"
            vector_store_exists = vector_store_path.exists()
            
            # Check if there are files in the input directory
            has_input_files = self._has_input_files()
            
            # Process files if they exist in the input directory
            if has_input_files:
                logger.info("Processing files in input directory...")
                
                # Step 1: Load documents from the input directory
                # This step finds and loads all supported documents
                documents = self.load_documents()
                # Abort if no documents were loaded
                if not documents:
                    logger.error("No documents loaded. Pipeline aborted.")
                    return False
                
                # Step 2: Clean and preprocess the loaded documents
                # This step normalizes and improves text quality
                cleaned_documents = self.clean_documents(documents)
                # Abort if document cleaning failed
                if not cleaned_documents:
                    logger.error("Document cleaning failed. Pipeline aborted.")
                    return False
                
                # Step 3: Chunk the cleaned documents into smaller pieces
                # This step creates appropriately sized chunks for retrieval
                chunked_documents = self.chunk_documents(cleaned_documents)
                # Abort if document chunking failed
                if not chunked_documents:
                    logger.error("Document chunking failed. Pipeline aborted.")
                    return False
                
                # Step 4: Embed chunks and store them in the vector database
                # This step converts text chunks to vector embeddings for retrieval
                vector_store = self.embed_chunks(chunked_documents)
                
                # Step 5: Save the processed outputs to files
                # This step creates both cleaned and chunked text files
                self.save_outputs(chunked_documents)
                
                # Clean up input folder after processing
                self._cleanup_input_folder()
                
                processing_time = time.time() - start_time
                logger.info(f"Processing completed in {processing_time:.2f} seconds")
            else:
                # If no input files but vector store exists, load it for retrieval
                if vector_store_exists:
                    logger.info("No input files found. Loading existing vector store for retrieval...")
                    self.vector_store = VectorStore.load(vector_store_path)
                    logger.info(f"Loaded vector store with {len(self.vector_store.documents)} documents")
                else:
                    # No input files and no vector store
                    logger.error("No input files found and no existing vector store. Pipeline aborted.")
                    return False
            
            # Run retrieval if a query is provided
            if query:
                logger.info(f"Running retrieval with query: '{query}'")
                retrieval_results = self.retrieve(
                    query=query,
                    reranking_enabled=self.config.reranking_enabled,
                    hybrid_search_enabled=self.config.hybrid_search_enabled,
                    hybrid_search_weight=self.config.hybrid_search_weight
                )
                logger.info(f"Retrieved {len(retrieval_results)} documents for query: '{query}' with enhanced retrieval settings")
            
            # Calculate and log performance metrics
            total_time = time.time() - start_time
            # Log successful completion with summary information
            logger.info("RAG Pipeline completed successfully!")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Outputs saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            # Handle any unexpected errors that occur during pipeline execution
            # Log both the error message and the full traceback for debugging
            logger.error(f"Pipeline failed with error: {e}")
            logger.exception("Error traceback:")
            return False


def main():
    """
    Main function to run the RAG pipeline from the command line.
    
    This function provides a command-line interface for the RAG pipeline, allowing users
    to specify input and output directories, embedding options, and control logging verbosity.
    It handles argument parsing, pipeline initialization, execution, and appropriate exit codes.
    
    Command-line arguments:
        --input, -i: Input directory containing documents (default: 'input')
        --output, -o: Output directory for processed files (default: 'output')
        --chunk-size: Size of text chunks in characters (default: 600)
        --chunk-overlap: Overlap between chunks in characters (default: 150)
        --embedding-model: Name of the embedding model to use (default: 'all-MiniLM-L6-v2')
        --embedding-provider: Provider for embeddings (default: 'sentence_transformers')
        --embedding-dim: Dimension of embedding vectors (default: 384)
        --embedding-api-key: API key for commercial embedding providers
        --distance-metric: Similarity metric for vector comparison (default: 'cosine')
        --verbose, -v: Enable verbose (DEBUG level) logging
    
    Exit codes:
        0: Pipeline completed successfully
        1: Pipeline failed (see logs for details)
    """
    # Set up command-line argument parser with descriptive help text
    parser = argparse.ArgumentParser(description="Simple RAG Pipeline - Process documents for retrieval")
    
    # Define command-line arguments with short and long forms
    # Input/output settings
    parser.add_argument('--input', '-i', default='input', 
                       help='Input directory containing documents (default: input)')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory for processed files (default: output)')
    
    # Chunking settings
    parser.add_argument('--chunk-size', type=int, default=600,
                       help='Size of text chunks in characters (default: 600)')
    parser.add_argument('--chunk-overlap', type=int, default=150,
                       help='Overlap between chunks in characters (default: 150)')
    
    # Embedding settings
    parser.add_argument('--embedding-model', default='nomic-embed-text',
                       help='Name of the embedding model to use (default: nomic-embed-text). '
                            'For local Nomic models, use format "local:/path/to/model"')
    parser.add_argument('--embedding-provider', default='nomic',
                       choices=['sentence_transformers', 'nomic', 'openai'],
                       help='Provider for embeddings (default: nomic)')
    parser.add_argument('--embedding-dim', type=int, default=768,
                       help='Dimension of embedding vectors (default: 768)')
    parser.add_argument('--embedding-api-key', default=None,
                       help='API key for commercial embedding providers')
    
    # Vector database settings
    parser.add_argument('--distance-metric', default='cosine',
                       choices=['cosine', 'euclidean', 'dot'],
                       help='Similarity metric for vector comparison (default: cosine)')
    
    # Retriever settings
    parser.add_argument('--retriever-top-k', type=int, default=5,
                       help='Number of results to return in retrieval (default: 5)')
    parser.add_argument('--retriever-threshold', type=float, default=0.0,
                       help='Minimum similarity threshold for retrieval results (default: 0.0)')
    parser.add_argument('--reranking-enabled', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable reranking for improved semantic results (default: True)')
    parser.add_argument('--hybrid-search-enabled', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable hybrid search combining semantic and keyword search (default: True)')
    parser.add_argument('--hybrid-search-weight', type=float, default=0.7,
                       help='Weight for balancing semantic vs keyword search (0-1) (default: 0.7)')
    
    # Retrieval settings
    parser.add_argument('--query', '-q', type=str,
                       help='Run retrieval with this query after processing')
    
    # Logging settings
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Configure logging based on verbosity flag
    # By default, logging is set to INFO level, but with --verbose it's set to DEBUG
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Create a configuration object with the command-line arguments
    config = RAGPipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        # Chunking settings
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        # Embedding settings
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        embedding_dim=args.embedding_dim,
        embedding_api_key=args.embedding_api_key,
        # Vector database settings
        distance_metric=args.distance_metric,
        # Retriever settings
        retriever_top_k=args.retriever_top_k,
        retriever_similarity_threshold=args.retriever_threshold,
        reranking_enabled=args.reranking_enabled,
        hybrid_search_enabled=args.hybrid_search_enabled,
        hybrid_search_weight=args.hybrid_search_weight
    )
    
    # Initialize the RAG pipeline with the configuration
    pipeline = RAGPipeline(config=config)
    
    # Execute the pipeline and capture the success/failure status
    logger.info(f"Running pipeline with input={args.input}, output={args.output}")
    logger.info(f"Using embedding model: {args.embedding_model} ({args.embedding_provider})")
    
    # Run with query if provided
    if args.query:
        logger.info(f"Will run retrieval with query: '{args.query}'")
    
    success = pipeline.run(query=args.query)
    
    # Handle the pipeline result with appropriate user feedback and exit codes
    if success:
        # Print user-friendly success message with emoji for visibility
        print("\n🎉 RAG Pipeline completed successfully!")
        print(f"📁 Check outputs in: {args.output}")
        # Exit with code 0 to indicate success to calling processes
        sys.exit(0)
    else:
        # Print user-friendly error message with emoji for visibility
        print("\n❌ RAG Pipeline failed!")
        print("📋 Check logs for details")
        # Exit with code 1 to indicate failure to calling processes
        sys.exit(1)


if __name__ == "__main__":
    main()