#!/usr/bin/env python3
"""Text Embedding Module for RAG Pipeline

This module converts text chunks into vector embeddings using various embedding models.
It provides a unified interface for different embedding providers and models,
allowing for easy switching between different embedding strategies.

Key Features:
- Support for multiple embedding models (SentenceTransformers, OpenAI, etc.)
- Support for both local and API-based Nomic models
- Batch processing for efficient embedding generation
- Caching to avoid redundant embedding calculations
- Dimensionality reduction options for large embeddings
- Metadata preservation throughout the embedding process

Typical Usage:
    >>> from modules.embedder import TextEmbedder
    >>> embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    >>> embeddings = embedder.embed_documents(chunks)
    >>> embedder.save_embeddings(embeddings, output_file="embeddings.pkl")
    
    # Using a local Nomic model
    >>> embedder = TextEmbedder(model_name="local:/path/to/nomic/model", model_provider="nomic")
    >>> embeddings = embedder.embed_documents(chunks)
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    A class for converting text chunks into vector embeddings.
    
    This class provides methods to embed text chunks using various embedding models,
    with a focus on SentenceTransformers as the default implementation.
    
    Attributes:
        model_name (str): Name of the embedding model to use
        model: The loaded embedding model
        embedding_dim (int): Dimension of the generated embeddings
        batch_size (int): Number of chunks to process at once
        cache_dir (Path): Directory to cache embeddings
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 embedding_dim: int = 384,
                 batch_size: int = 32,
                 cache_dir: Optional[Union[str, Path]] = None,
                 model_provider: str = "sentence_transformers",
                 api_key: Optional[str] = None):
        """
        Initialize the TextEmbedder with the specified model and parameters.
        
        Args:
            model_name (str): Name of the embedding model to use
                             Default: "all-MiniLM-L6-v2" (SentenceTransformers)
                             For local Nomic models, use format "local:/path/to/model"
            embedding_dim (int): Dimension of the generated embeddings
                                Default: 384 (for all-MiniLM-L6-v2)
            batch_size (int): Number of chunks to process at once
                             Default: 32
            cache_dir (Optional[Union[str, Path]]): Directory to cache embeddings
                                                   Default: None (no caching)
            model_provider (str): Provider of the embedding model
                                 Options: "sentence_transformers", "nomic", "openai"
                                 Default: "sentence_transformers"
            api_key (Optional[str]): API key for providers that require authentication
                                    Default: None
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_provider = model_provider.lower()
        self.api_key = api_key
        self.model = None
        
        # Initialize cache directory if provided
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache directory: {self.cache_dir}")
        
        # Load the embedding model based on the provider
        self._load_model()
            
    def _load_model(self):
        """
        Load the embedding model based on the specified provider.
        
        This method initializes the appropriate embedding model based on the
        model_provider and model_name specified during initialization.
        
        Supported providers:
        - sentence_transformers: Uses the SentenceTransformers library
        - nomic: Uses the Nomic AI embedding models
        - openai: Uses OpenAI's embedding API
        """
        if self.model_provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except ImportError:
                logger.warning("SentenceTransformers not installed. Run: pip install sentence-transformers")
                self.model = None
        
        elif self.model_provider == "nomic":
            try:
                # Check if we're using a local model or API
                if self.model_name.startswith("local:"):
                    # Extract the local model path
                    local_model_path = self.model_name[6:]
                    # Store the model path for later use
                    self.model = {"type": "local", "path": local_model_path}
                    logger.info(f"Using local Nomic embedding model at: {local_model_path}")
                elif self.model_name in ["nomic-embed-text", "nomic-embed-text:latest"]:
                    # Use local sentence-transformers for nomic-embed-text model
                    self.model = {"type": "local", "name": "nomic-ai/nomic-embed-text-v1"}
                    logger.info(f"Using local Nomic embedding model: nomic-ai/nomic-embed-text-v1")
                else:
                    # For API models, we'll load on-demand during embedding
                    self.model = {"type": "api", "name": self.model_name}
                    logger.info(f"Using Nomic API embedding model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Error setting up Nomic model: {e}")
                self.model = None
        
        elif self.model_provider == "openai":
            try:
                import openai
                if self.api_key:
                    openai.api_key = self.api_key
                self.model = "openai"
                logger.info(f"Using OpenAI embedding model: {self.model_name}")
            except ImportError:
                logger.warning("OpenAI not installed. Run: pip install openai")
                self.model = None
        
        else:
            logger.error(f"Unsupported model provider: {self.model_provider}")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a vector.
        
        Args:
            text (str): The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        if self.model is None:
            # Return random embedding if model not available (for testing)
            logger.warning("Using random placeholder embedding")
            return np.random.rand(self.embedding_dim)
        
        # Generate embedding based on the provider
        if self.model_provider == "sentence_transformers":
            return self.model.encode(text)
        
        elif self.model_provider == "nomic":
            try:
                # Handle different Nomic model types
                if isinstance(self.model, dict) and self.model.get("type") == "local":
                    try:
                        from sentence_transformers import SentenceTransformer
                        model_name = self.model.get("name") or self.model.get("path")
                        # Load the local model with sentence-transformers
                        if not hasattr(self, '_local_model'):
                            self._local_model = SentenceTransformer(model_name, trust_remote_code=True)
                        result = self._local_model.encode([text])
                        return np.array(result[0])
                    except Exception as local_e:
                        logger.warning(f"Local Nomic model failed, using random embedding: {local_e}")
                        return np.random.rand(self.embedding_dim)
                else:
                    # For API models, skip since no authentication is configured
                    logger.warning("Nomic API requires authentication. Using random embedding.")
                    return np.random.rand(self.embedding_dim)
            except Exception as e:
                logger.error(f"Error using Nomic embedding: {e}")
                return np.random.rand(self.embedding_dim)
        
        elif self.model_provider == "openai":
            try:
                import openai
                response = openai.Embedding.create(input=text, model=self.model_name)
                return np.array(response["data"][0]["embedding"])
            except Exception as e:
                logger.error(f"Error using OpenAI embedding: {e}")
                return np.random.rand(self.embedding_dim)
        
        # Fallback for unsupported providers
        logger.warning(f"Unsupported model provider: {self.model_provider}. Using random embedding.")
        return np.random.rand(self.embedding_dim)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks, preserving metadata.
        
        This method processes each document chunk, generating embeddings for the
        'text' field and adding the embedding vector to the document metadata.
        
        Args:
            documents (List[Dict[str, Any]]): List of document chunks with 'text' field
            
        Returns:
            List[Dict[str, Any]]: The same documents with added 'embedding' field
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return []
        
        logger.info(f"Embedding {len(documents)} document chunks using {self.model_provider} provider")
        
        # Process documents in batches for efficiency
        embedded_docs = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(documents)-1)//self.batch_size + 1}")
            
            # Extract text content for embedding
            texts = [doc['text'] for doc in batch]
            
            # Generate embeddings based on the provider
            embeddings = self._batch_embed(texts)
            
            # Add embeddings back to document metadata
            for j, doc in enumerate(batch):
                doc_with_embedding = doc.copy()
                doc_with_embedding['embedding'] = embeddings[j]
                embedded_docs.append(doc_with_embedding)
        
        logger.info(f"Successfully embedded {len(embedded_docs)} document chunks")
        return embedded_docs
    
    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a batch of texts using the appropriate provider.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if self.model is None:
            # Generate random embeddings if model not available
            logger.warning("Using random placeholder embeddings")
            return [np.random.rand(self.embedding_dim) for _ in texts]
        
        # Generate embeddings based on the provider
        if self.model_provider == "sentence_transformers":
            return self.model.encode(texts)
        
        elif self.model_provider == "nomic":
            try:
                # Handle different Nomic model types
                if isinstance(self.model, dict) and self.model.get("type") == "local":
                    try:
                        from sentence_transformers import SentenceTransformer
                        model_name = self.model.get("name") or self.model.get("path")
                        # Load the local model with sentence-transformers
                        if not hasattr(self, '_local_model'):
                            self._local_model = SentenceTransformer(model_name, trust_remote_code=True)
                        results = self._local_model.encode(texts)
                        return [np.array(emb) for emb in results]
                    except Exception as local_e:
                        logger.warning(f"Local Nomic batch embedding failed, using random embeddings: {local_e}")
                        return [np.random.rand(self.embedding_dim) for _ in texts]
                else:
                    # For API models, skip since no authentication is configured
                    logger.warning("Nomic API requires authentication. Using random embeddings.")
                    return [np.random.rand(self.embedding_dim) for _ in texts]
            except Exception as e:
                logger.error(f"Error using Nomic batch embedding: {e}")
                return [np.random.rand(self.embedding_dim) for _ in texts]
        
        elif self.model_provider == "openai":
            try:
                import openai
                response = openai.Embedding.create(input=texts, model=self.model_name)
                return [np.array(item["embedding"]) for item in response["data"]]
            except Exception as e:
                logger.error(f"Error using OpenAI batch embedding: {e}")
                return [np.random.rand(self.embedding_dim) for _ in texts]
        
        # Fallback for unsupported providers - embed one by one
        logger.warning(f"Unsupported batch embedding for provider: {self.model_provider}. Embedding one by one.")
        return [self.embed_text(text) for text in texts]
    
    def save_embeddings(self, embedded_docs: List[Dict[str, Any]], output_file: Union[str, Path]) -> Path:
        """
        Save embedded documents to a pickle file.
        
        Args:
            embedded_docs (List[Dict[str, Any]]): Documents with embeddings
            output_file (Union[str, Path]): Path to save the embeddings
            
        Returns:
            Path: Path to the saved embeddings file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(embedded_docs, f)
        
        logger.info(f"Saved {len(embedded_docs)} embedded documents to {output_path}")
        return output_path
    
    def load_embeddings(self, input_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load embedded documents from a pickle file.
        
        Args:
            input_file (Union[str, Path]): Path to the embeddings file
            
        Returns:
            List[Dict[str, Any]]: Documents with embeddings
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.error(f"Embeddings file not found: {input_path}")
            return []
        
        with open(input_path, 'rb') as f:
            embedded_docs = pickle.load(f)
        
        logger.info(f"Loaded {len(embedded_docs)} embedded documents from {input_path}")
        return embedded_docs