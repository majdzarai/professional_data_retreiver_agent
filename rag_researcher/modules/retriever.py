#!/usr/bin/env python3
"""
Retriever Module for RAG Pipeline

This module provides document retrieval capabilities for the RAG pipeline,
allowing for efficient similarity search of embedded document chunks using
cosine similarity.

Key Features:
- Configurable similarity search with cosine similarity
- Adjustable top-k results parameter
- Similarity threshold filtering
- Metadata filtering support
- Integration with VectorStore for embeddings

Typical Usage:
    >>> from modules.retriever import Retriever
    >>> from modules.vector_store import VectorStore
    >>> from modules.embedder import TextEmbedder
    >>> retriever = Retriever(vector_store, embedder)
    >>> results = retriever.retrieve("What is machine learning?", top_k=5)
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    """
    An advanced retriever for finding semantically relevant document chunks.
    
    This class provides methods to retrieve document chunks based on semantic similarity,
    with configurable parameters for controlling the number of results, similarity threshold,
    and advanced retrieval techniques like reranking and hybrid search.
    
    Attributes:
        vector_store: The vector store containing document embeddings
        embedder: The text embedder for converting queries to embeddings
        top_k (int): Default number of results to return
        similarity_threshold (float): Minimum similarity score for results (0-1)
        reranking_enabled (bool): Whether to use reranking for improved results
        hybrid_search_enabled (bool): Whether to use hybrid search (semantic + keyword)
        hybrid_search_weight (float): Weight for balancing semantic vs keyword search (0-1)
    """
    
    def __init__(self, 
                 vector_store,
                 embedder,
                 top_k: int = 5,
                 similarity_threshold: float = 0.0,
                 reranking_enabled: bool = True,
                 hybrid_search_enabled: bool = True,
                 hybrid_search_weight: float = 0.7):
        """
        Initialize the Retriever with vector store and embedder.
        
        Args:
            vector_store: Vector store containing document embeddings
            embedder: Text embedder for converting queries to embeddings
            top_k (int): Default number of results to return
                        Default: 5
            similarity_threshold (float): Minimum similarity score (0-1)
                                        Default: 0.0 (no threshold)
            reranking_enabled (bool): Whether to use reranking for improved results
                                    Default: True
            hybrid_search_enabled (bool): Whether to use hybrid search (semantic + keyword)
                                        Default: True
            hybrid_search_weight (float): Weight for balancing semantic vs keyword search (0-1)
                                        Default: 0.7 (70% semantic, 30% keyword)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranking_enabled = reranking_enabled
        self.hybrid_search_enabled = hybrid_search_enabled
        self.hybrid_search_weight = hybrid_search_weight
        
        # Try to import optional dependencies
        self._init_optional_dependencies()
        
        logger.info(f"Initialized Retriever with top_k={top_k}, similarity_threshold={similarity_threshold}, "
                   f"reranking_enabled={reranking_enabled}, hybrid_search_enabled={hybrid_search_enabled}, "
                   f"hybrid_search_weight={hybrid_search_weight}")
    
    def _init_optional_dependencies(self):
        """
        Initialize optional dependencies for advanced retrieval features.
        """
        self.reranker = None
        self.keyword_search_available = False
        
        # Try to import reranking libraries if reranking is enabled
        if self.reranking_enabled:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                # Initialize a simple cross-encoder reranker
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # A good default reranker model
                try:
                    self.reranker = {
                        "tokenizer": AutoTokenizer.from_pretrained(model_name),
                        "model": AutoModelForSequenceClassification.from_pretrained(model_name)
                    }
                    if torch.cuda.is_available():
                        self.reranker["model"] = self.reranker["model"].to("cuda")
                    logger.info(f"Initialized reranker model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load reranker model: {e}")
                    self.reranking_enabled = False
            except ImportError:
                logger.warning("Transformers or torch not installed. Reranking disabled.")
                self.reranking_enabled = False
        
        # Try to import keyword search libraries if hybrid search is enabled
        if self.hybrid_search_enabled:
            try:
                import nltk
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                
                # Download necessary NLTK resources
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                self.keyword_search_available = True
                logger.info("Initialized keyword search capabilities for hybrid retrieval")
            except ImportError:
                logger.warning("NLTK not installed. Hybrid search disabled.")
                self.hybrid_search_enabled = False
    
    def _perform_keyword_search(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search on documents.
        
        Args:
            query (str): The query text
            documents (List[Dict[str, Any]]): The documents to search
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Documents ranked by keyword relevance
        """
        if not self.keyword_search_available or not documents:
            return []
        
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from collections import Counter
            
            # Tokenize and normalize query
            stop_words = set(stopwords.words('english'))
            query_tokens = [token.lower() for token in word_tokenize(query) 
                           if token.isalnum() and token.lower() not in stop_words]
            
            # Calculate TF-IDF style scores for each document
            results = []
            for doc in documents:
                doc_text = doc.get('text', '')
                doc_tokens = [token.lower() for token in word_tokenize(doc_text) 
                             if token.isalnum() and token.lower() not in stop_words]
                
                # Count token occurrences
                doc_counter = Counter(doc_tokens)
                
                # Calculate score based on query token presence and frequency
                score = 0
                for token in query_tokens:
                    if token in doc_counter:
                        # Simple TF scoring
                        score += doc_counter[token] / len(doc_tokens) if len(doc_tokens) > 0 else 0
                
                # Normalize by query length
                score = score / len(query_tokens) if len(query_tokens) > 0 else 0
                
                # Create result with keyword score
                result = doc.copy()
                result['keyword_score'] = score
                results.append(result)
            
            # Sort by keyword score and take top_k
            results = sorted(results, key=lambda x: x.get('keyword_score', 0), reverse=True)[:top_k]
            return results
        
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank results using a cross-encoder model for improved relevance.
        
        Args:
            query (str): The query text
            results (List[Dict[str, Any]]): The results to rerank
            top_k (int): Number of results to return after reranking
            
        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        if not self.reranking_enabled or not self.reranker or not results:
            return results[:top_k] if results else []
        
        try:
            import torch
            
            # Prepare pairs for reranking
            pairs = [(query, doc.get('text', '')) for doc in results]
            
            # Tokenize pairs
            features = self.reranker["tokenizer"](pairs, padding=True, truncation=True, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                features = {k: v.to("cuda") for k, v in features.items()}
            
            # Get reranking scores
            with torch.no_grad():
                scores = self.reranker["model"](**features).logits.flatten()
            
            # Convert to numpy for easier handling
            scores = scores.cpu().numpy()
            
            # Add reranking scores to results
            for i, score in enumerate(scores):
                results[i]['rerank_score'] = float(score)
            
            # Sort by reranking score and take top_k
            reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)[:top_k]
            
            logger.info(f"Reranked {len(results)} results to {len(reranked_results)} results")
            return reranked_results
        
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results[:top_k] if results else []
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None, 
                similarity_threshold: Optional[float] = None,
                filter_dict: Optional[Dict[str, Any]] = None,
                reranking_enabled: Optional[bool] = None,
                hybrid_search_enabled: Optional[bool] = None,
                hybrid_search_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve semantically relevant document chunks based on query.
        
        Args:
            query (str): The query text
            top_k (Optional[int]): Number of results to return, overrides default
            similarity_threshold (Optional[float]): Minimum similarity score (0-1), overrides default
            filter_dict (Optional[Dict[str, Any]]): Metadata filters to apply
            reranking_enabled (Optional[bool]): Whether to use reranking, overrides default
            hybrid_search_enabled (Optional[bool]): Whether to use hybrid search, overrides default
            hybrid_search_weight (Optional[float]): Weight for hybrid search, overrides default
            
        Returns:
            List[Dict[str, Any]]: The most relevant document chunks with similarity scores
        """
        # Use provided parameters or fall back to instance defaults
        top_k = top_k if top_k is not None else self.top_k
        similarity_threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        reranking_enabled = reranking_enabled if reranking_enabled is not None else self.reranking_enabled
        hybrid_search_enabled = hybrid_search_enabled if hybrid_search_enabled is not None else self.hybrid_search_enabled
        hybrid_search_weight = hybrid_search_weight if hybrid_search_weight is not None else self.hybrid_search_weight
        
        # Increase top_k for initial retrieval if we're going to rerank
        initial_top_k = top_k * 3 if reranking_enabled else top_k
        
        logger.info(f"Retrieving documents for query: '{query}' (top_k={top_k}, threshold={similarity_threshold}, "
                   f"reranking={reranking_enabled}, hybrid={hybrid_search_enabled})")
        
        # Convert query to embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Retrieve similar documents from vector store using semantic search
        semantic_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=initial_top_k,
            filter_dict=filter_dict
        )
        
        # Apply hybrid search if enabled
        if hybrid_search_enabled and self.keyword_search_available and semantic_results:
            # Get all documents from vector store for keyword search
            # In a production system, you would implement a more efficient keyword index
            all_docs = self.vector_store.documents
            
            # Perform keyword search
            keyword_results = self._perform_keyword_search(query, all_docs, initial_top_k)
            
            # Combine semantic and keyword results with weighting
            if keyword_results:
                # Create a map of document IDs to their semantic scores
                semantic_scores = {}
                for doc in semantic_results:
                    # Use a unique identifier for each document
                    doc_id = doc.get('metadata', {}).get('source', '') + str(doc.get('metadata', {}).get('chunk_id', ''))
                    semantic_scores[doc_id] = doc.get('similarity_score', 0)
                
                # Create a map of document IDs to their keyword scores
                keyword_scores = {}
                for doc in keyword_results:
                    doc_id = doc.get('metadata', {}).get('source', '') + str(doc.get('metadata', {}).get('chunk_id', ''))
                    keyword_scores[doc_id] = doc.get('keyword_score', 0)
                
                # Combine all unique documents
                all_doc_ids = set(list(semantic_scores.keys()) + list(keyword_scores.keys()))
                combined_results = []
                
                for doc_id in all_doc_ids:
                    # Find the document in either result set
                    doc = None
                    for d in semantic_results + keyword_results:
                        curr_id = d.get('metadata', {}).get('source', '') + str(d.get('metadata', {}).get('chunk_id', ''))
                        if curr_id == doc_id:
                            doc = d.copy()
                            break
                    
                    if doc:
                        # Calculate combined score
                        semantic_score = semantic_scores.get(doc_id, 0)
                        keyword_score = keyword_scores.get(doc_id, 0)
                        
                        # Weighted combination
                        combined_score = (hybrid_search_weight * semantic_score + 
                                         (1 - hybrid_search_weight) * keyword_score)
                        
                        doc['similarity_score'] = combined_score
                        doc['semantic_score'] = semantic_score
                        doc['keyword_score'] = keyword_score
                        combined_results.append(doc)
                
                # Sort by combined score
                combined_results = sorted(combined_results, key=lambda x: x.get('similarity_score', 0), reverse=True)
                results = combined_results[:initial_top_k]
                logger.info(f"Applied hybrid search, combined {len(semantic_results)} semantic and "
                           f"{len(keyword_results)} keyword results into {len(results)} results")
            else:
                results = semantic_results
        else:
            results = semantic_results
        
        # Apply similarity threshold if specified
        if similarity_threshold > 0:
            results = [doc for doc in results if doc.get('similarity_score', 0) >= similarity_threshold]
            logger.info(f"Applied similarity threshold {similarity_threshold}, {len(results)} results remain")
        
        # Apply reranking if enabled and we have results
        if reranking_enabled and self.reranker and results:
            results = self._rerank_results(query, results, top_k)
        else:
            # Just take top_k results if no reranking
            results = results[:top_k]
        
        # Log retrieval results
        if results:
            logger.info(f"Retrieved {len(results)} relevant documents")
            # Log top result similarity score for debugging
            if results[0].get('similarity_score') is not None:
                logger.info(f"Top result similarity score: {results[0]['similarity_score']:.4f}")
        else:
            logger.warning("No relevant documents found for query")
        
        return results
    
    def retrieve_and_format(self, 
                          query: str, 
                          top_k: Optional[int] = None,
                          similarity_threshold: Optional[float] = None,
                          filter_dict: Optional[Dict[str, Any]] = None,
                          include_metadata: bool = True,
                          reranking_enabled: Optional[bool] = None,
                          hybrid_search_enabled: Optional[bool] = None,
                          hybrid_search_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve and format document chunks for display or further processing.
        
        This method retrieves semantically relevant documents and formats them in a more
        user-friendly way, optionally including or excluding metadata. It supports advanced
        retrieval features like reranking and hybrid search for improved semantic relevance.
        
        Args:
            query (str): The query text
            top_k (Optional[int]): Number of results to return, overrides default
            similarity_threshold (Optional[float]): Minimum similarity score (0-1), overrides default
            filter_dict (Optional[Dict[str, Any]]): Metadata filters to apply
            include_metadata (bool): Whether to include metadata in results
            reranking_enabled (Optional[bool]): Whether to use reranking for improved results
            hybrid_search_enabled (Optional[bool]): Whether to use hybrid search (semantic + keyword)
            hybrid_search_weight (Optional[float]): Weight for balancing semantic vs keyword search (0-1)
            
        Returns:
            List[Dict[str, Any]]: Formatted document chunks with text and optional metadata
        """
        # Retrieve documents with enhanced semantic search capabilities
        results = self.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_dict=filter_dict,
            reranking_enabled=reranking_enabled,
            hybrid_search_enabled=hybrid_search_enabled,
            hybrid_search_weight=hybrid_search_weight
        )
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results):
            formatted_doc = {
                'id': i + 1,
                'text': doc.get('text', ''),
                'similarity_score': doc.get('similarity_score', 0)
            }
            
            # Include metadata if requested
            if include_metadata and 'metadata' in doc:
                formatted_doc['metadata'] = doc['metadata']
            
            formatted_results.append(formatted_doc)
        
        return formatted_results