#!/usr/bin/env python3
"""
Vector Database Module for RAG Pipeline

This module provides vector storage and retrieval capabilities for the RAG pipeline,
allowing for efficient similarity search of embedded document chunks.

Key Features:
- In-memory vector storage with numpy for development and testing
- Extensible interface for different vector database backends
- Similarity search with configurable distance metrics
- Metadata filtering for advanced retrieval
- Batch operations for efficient processing

Typical Usage:
    >>> from modules.vector_store import VectorStore
    >>> vector_store = VectorStore()
    >>> vector_store.add_documents(embedded_documents)
    >>> results = vector_store.similarity_search(query_embedding, top_k=5)
"""

import os
import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    A simple in-memory vector database for storing and retrieving document embeddings.
    
    This class provides methods to store document embeddings and perform similarity
    searches using various distance metrics. It's designed as a lightweight solution
    for development and testing, with the ability to extend to other vector database
    backends.
    
    Attributes:
        documents (List[Dict]): List of documents with embeddings
        embedding_matrix (np.ndarray): Matrix of document embeddings for fast search
        distance_metric (str): Distance metric for similarity search
    """
    
    def __init__(self, distance_metric: str = "cosine"):
        """
        Initialize the VectorStore with the specified distance metric.
        
        Args:
            distance_metric (str): Distance metric for similarity search
                                  Options: "cosine", "euclidean", "dot"
                                  Default: "cosine"
        """
        self.documents = []
        self.embedding_matrix = None
        self.distance_metric = distance_metric
        logger.info(f"Initialized VectorStore with {distance_metric} distance metric")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents (List[Dict[str, Any]]): Documents with 'embedding' field
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        # Validate that documents have embeddings
        for doc in documents:
            if 'embedding' not in doc:
                logger.error(f"Document missing 'embedding' field: {doc.get('chunk_id', 'unknown')}")
                return
        
        # Add documents to the store
        self.documents.extend(documents)
        
        # Update the embedding matrix for fast search
        self._update_embedding_matrix()
        
        logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")
    
    def _update_embedding_matrix(self) -> None:
        """
        Update the embedding matrix for fast similarity search.
        
        This method extracts embeddings from all documents and stacks them
        into a single numpy matrix for efficient vector operations.
        """
        if not self.documents:
            logger.warning("No documents in vector store to update embedding matrix")
            self.embedding_matrix = None
            return
        
        # Extract embeddings from documents
        embeddings = [doc['embedding'] for doc in self.documents]
        
        # Stack embeddings into a matrix
        self.embedding_matrix = np.vstack(embeddings)
        
        logger.debug(f"Updated embedding matrix: {self.embedding_matrix.shape}")
    
    def similarity_search(self, 
                          query_embedding: np.ndarray, 
                          top_k: int = 5, 
                          filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find the most similar documents to a query embedding.
        
        Args:
            query_embedding (np.ndarray): The query embedding vector
            top_k (int): Number of results to return
                        Default: 5
            filter_dict (Optional[Dict[str, Any]]): Metadata filters to apply
                                                  Default: None
            
        Returns:
            List[Dict[str, Any]]: The most similar documents with similarity scores
        """
        if not self.documents or self.embedding_matrix is None:
            logger.warning("No documents in vector store for similarity search")
            return []
        
        # Calculate similarity scores based on distance metric
        if self.distance_metric == "cosine":
            # Normalize vectors for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Calculate cosine similarity
            matrix_norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            normalized_matrix = self.embedding_matrix / np.maximum(matrix_norms, 1e-10)
            scores = np.dot(normalized_matrix, query_embedding)
            
        elif self.distance_metric == "euclidean":
            # Calculate negative euclidean distance (higher is more similar)
            scores = -np.linalg.norm(self.embedding_matrix - query_embedding, axis=1)
            
        elif self.distance_metric == "dot":
            # Calculate dot product
            scores = np.dot(self.embedding_matrix, query_embedding)
            
        else:
            logger.error(f"Unsupported distance metric: {self.distance_metric}")
            return []
        
        # Apply metadata filters if provided
        if filter_dict:
            filtered_indices = self._apply_filters(filter_dict)
            scores = scores[filtered_indices]
            doc_indices = filtered_indices
        else:
            doc_indices = np.arange(len(self.documents))
        
        # Get top-k results
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Prepare results with documents and scores
        results = []
        for idx in top_indices:
            doc_idx = doc_indices[idx]
            doc = self.documents[doc_idx].copy()
            doc['similarity_score'] = float(scores[idx])  # Convert to Python float for JSON serialization
            results.append(doc)
        
        logger.info(f"Found {len(results)} similar documents for query")
        return results
    
    def _apply_filters(self, filter_dict: Dict[str, Any]) -> np.ndarray:
        """
        Apply metadata filters to documents.
        
        Args:
            filter_dict (Dict[str, Any]): Metadata filters to apply
            
        Returns:
            np.ndarray: Indices of documents that match the filters
        """
        indices = []
        for i, doc in enumerate(self.documents):
            match = True
            for key, value in filter_dict.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                indices.append(i)
        
        return np.array(indices)
    
    def save(self, file_path: Union[str, Path]) -> Path:
        """
        Save the vector store to a file.
        
        Args:
            file_path (Union[str, Path]): Path to save the vector store
            
        Returns:
            Path: Path to the saved vector store file
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save documents and configuration
        data = {
            'documents': self.documents,
            'distance_metric': self.distance_metric
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved vector store with {len(self.documents)} documents to {output_path}")
        return output_path
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'VectorStore':
        """
        Load a vector store from a file.
        
        Args:
            file_path (Union[str, Path]): Path to the vector store file
            
        Returns:
            VectorStore: The loaded vector store
        """
        input_path = Path(file_path)
        
        if not input_path.exists():
            logger.error(f"Vector store file not found: {input_path}")
            return cls()
        
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new instance with the loaded configuration
        vector_store = cls(distance_metric=data.get('distance_metric', 'cosine'))
        vector_store.documents = data.get('documents', [])
        vector_store._update_embedding_matrix()
        
        logger.info(f"Loaded vector store with {len(vector_store.documents)} documents from {input_path}")
        return vector_store