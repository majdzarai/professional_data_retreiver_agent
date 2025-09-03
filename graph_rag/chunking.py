#!/usr/bin/env python3
"""
Text Chunking Module for Graph RAG Pipeline

This module provides intelligent text chunking capabilities specifically designed for
knowledge graph construction and retrieval-augmented generation (RAG) systems.
It handles document loading from multiple formats and implements sentence-aware
chunking to preserve semantic coherence.

Key Features:
- Multi-format document loading (TXT, MD, JSON)
- Sentence-aware chunking with configurable overlap
- Metadata preservation for traceability
- Human-readable output for debugging and validation
- Optimized for LLM processing and entity extraction

Typical Usage:
    >>> from graph_rag.chunking import load_documents, chunk_text, save_chunks_to_file
    >>> documents = load_documents("input_data")
    >>> for filename, text in documents:
    ...     chunks = chunk_text(text, chunk_size=400, overlap=50)
    ...     save_chunks_to_file(chunks, f"output/{filename}_chunks.txt")

Author: Graph RAG Pipeline
Version: 1.0
Compatibility: Python 3.7+
"""

import os
import json
import re
import hashlib
from datetime import datetime

def load_documents(input_folder="input_data"):
    """
    Loads and processes documents from multiple file formats for AML/KYC investigations.
    
    This function is the entry point for document ingestion in the Graph RAG pipeline.
    It supports text files (.txt), markdown files (.md), and JSON files (.json).
    For JSON files, it recursively extracts all text content regardless of structure.
    
    Args:
        input_folder (str): Path to the folder containing documents to process.
                           Defaults to "input_data" for standard pipeline structure.
    
    Returns:
        list: A list of tuples where each tuple contains:
              - filename (str): Original filename for traceability
              - text (str): Extracted text content ready for chunking
    
    Raises:
        FileNotFoundError: If input_folder doesn't exist
        PermissionError: If files cannot be read due to permissions
        json.JSONDecodeError: If JSON files are malformed (not handled currently)
    
    Example:
        >>> docs = load_documents("compliance_docs")
        >>> print(f"Loaded {len(docs)} documents")
        >>> filename, content = docs[0]
    """
    # Initialize empty list to store (filename, text) tuples
    documents = []

    # Iterate through all files in the specified input folder
    # Note: This does not recursively search subdirectories
    for file_name in os.listdir(input_folder):
        # Construct full file path for reading
        file_path = os.path.join(input_folder, file_name)
        
        # Handle plain text and markdown files
        # These are read directly as they contain human-readable text
        if file_name.endswith(".txt") or file_name.endswith(".md"):
            # Use UTF-8 encoding to handle international characters in compliance docs
            with open(file_path, "r", encoding="utf-8") as f:
                # Store filename for traceability and full text content
                documents.append((file_name, f.read()))

        # Handle JSON files which may contain structured compliance data
        elif file_name.endswith(".json"):
            # Load JSON data into Python object
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Recursive function to extract all text from nested JSON structures
            # This is crucial for compliance docs that may have complex nested data
            def extract_text(obj):
                """
                Recursively extracts text from any JSON structure.
                
                This handles various JSON patterns commonly found in compliance data:
                - Simple string values (names, addresses, descriptions)
                - Nested objects (customer profiles, transaction details)
                - Arrays of data (transaction lists, document references)
                - Mixed data types (converting numbers/booleans to empty strings)
                """
                if isinstance(obj, str):
                    # Add newline to separate text blocks for better readability
                    return obj + "\n"
                elif isinstance(obj, dict):
                    # Extract text from all dictionary values and join with spaces
                    # This flattens nested compliance data structures
                    return " ".join([extract_text(v) for v in obj.values()])
                elif isinstance(obj, list):
                    # Process each list item and join results
                    # Common for arrays of transactions, documents, etc.
                    return " ".join([extract_text(i) for i in obj])
                else:
                    # Handle non-text data (numbers, booleans, null) by ignoring
                    # This prevents errors while focusing on textual content
                    return ""
            
            # Apply text extraction and store result
            documents.append((file_name, extract_text(data)))

    return documents


def chunk_text(text, chunk_size=400, overlap=50, source_filename=None):
    """
    Intelligently splits text into overlapping chunks for KYC/AML Graph RAG processing.
    
    This function implements sentence-aware chunking with comprehensive audit metadata
    specifically designed for KYC/AML compliance investigations. It preserves semantic
    coherence while generating detailed traceability information required for regulatory
    audits and compliance reporting.
    
    The chunking strategy:
    1. Split text into sentences using regex pattern matching
    2. Group sentences until reaching the token limit (chunk_size)
    3. Create overlapping chunks to preserve context across boundaries
    4. Generate comprehensive audit metadata for compliance traceability
    
    Args:
        text (str): Input text to be chunked (from load_documents output)
        chunk_size (int): Maximum number of tokens (words) per chunk.
                         Default 400 is optimized for LLM context windows.
        overlap (int): Number of tokens to overlap between consecutive chunks.
                      Default 50 provides good context continuity.
        source_filename (str): Original filename for audit trail and traceability.
                              Used in compliance reporting and investigation tracking.
    
    Returns:
        list: List of dictionaries, each containing:
              - chunk_id (int): Sequential identifier for the chunk
              - text (str): The actual text content of the chunk
              - metadata (dict): Comprehensive audit metadata containing:
                  - source_file (str): Original document filename
                  - chunk_position (str): Position indicator (first/middle/last)
                  - start_sentence (int): Index of first sentence in chunk
                  - end_sentence (int): Index of last sentence in chunk
                  - token_count (int): Actual token count in the chunk
                  - character_count (int): Character count for size validation
                  - processing_timestamp (str): ISO timestamp for audit trail
                  - chunk_hash (str): Content hash for integrity verification
                  - overlap_with_previous (bool): Indicates if chunk has overlap
                  - compliance_flags (dict): KYC/AML specific metadata
    
    Example:
        >>> chunks = chunk_text("First sentence. Second sentence.", 
        ...                    chunk_size=5, source_filename="report.txt")
        >>> print(chunks[0]['metadata']['source_file'])
        >>> print(chunks[0]['metadata']['chunk_position'])
    
    Note:
        - Uses simple whitespace tokenization (may not align with LLM tokenizers)
        - Overlap calculation uses rough estimation (10 tokens per sentence)
        - Generates SHA-256 hash for content integrity verification
        - Includes compliance-specific metadata for audit requirements
    """
    # Split text into sentences using regex pattern
    # Pattern (?<=[.!?]) + matches one or more spaces after sentence-ending punctuation
    # This preserves sentence boundaries which is critical for semantic coherence
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Calculate total document statistics for audit metadata
    total_sentences = len(sentences)
    total_characters = len(text)
    processing_timestamp = datetime.now().isoformat()
    
    # Extract base filename for audit trail
    if source_filename:
        base_filename = os.path.basename(source_filename)
    else:
        base_filename = "unknown_source"

    # Initialize variables for chunk processing
    chunks = []  # Final list of processed chunks
    current_chunk = []  # Sentences being accumulated for current chunk
    current_length = 0  # Running count of tokens in current chunk
    chunk_index = 0  # Sequential identifier for chunks

    # Process each sentence and build chunks
    for i, sentence in enumerate(sentences):
        # Tokenize sentence using simple whitespace splitting
        # Note: This may not match LLM tokenization exactly
        tokens = sentence.split()
        
        # Check if adding this sentence would exceed chunk size limit
        if current_length + len(tokens) > chunk_size:
            # Current chunk is full - save it before starting new one
            chunk_text = " ".join(current_chunk)
            
            # Generate comprehensive audit metadata
            chunk_metadata = {
                # Core identification
                "source_file": base_filename,
                "chunk_position": "first" if chunk_index == 0 else "middle",
                
                # Sentence boundaries for traceability
                "start_sentence": i - len(current_chunk),
                "end_sentence": i - 1,
                
                # Size metrics for validation
                "token_count": current_length,
                "character_count": len(chunk_text),
                
                # Audit trail information
                "processing_timestamp": processing_timestamp,
                "chunk_hash": hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16],
                
                # Processing context
                "overlap_with_previous": chunk_index > 0,
                "total_chunks_in_document": "TBD",  # Will be updated after processing
                
                # KYC/AML compliance metadata
                "compliance_flags": {
                    "document_type": "financial_report" if "report" in base_filename.lower() else "unknown",
                    "contains_financial_data": any(term in chunk_text.lower() for term in ["revenue", "profit", "loss", "financial", "million", "billion", "eur", "usd", "$"]),
                    "contains_entity_names": any(char.isupper() for char in chunk_text if char.isalpha()),
                    "requires_review": len(chunk_text) > chunk_size * 4,  # Flag unusually long chunks
                    "processing_stage": "chunking",
                    "audit_ready": True
                }
            }
            
            chunks.append({
                "chunk_id": chunk_index,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            chunk_index += 1

            # Start new chunk with overlap from previous chunk
            # Overlap helps maintain context across chunk boundaries
            # Formula: overlap//10 assumes roughly 10 tokens per sentence
            overlap_sentences = current_chunk[-overlap//10:]
            current_chunk = overlap_sentences + [sentence]
            
            # Recalculate length including overlap and new sentence
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            # Sentence fits in current chunk - add it
            current_chunk.append(sentence)
            current_length += len(tokens)

    # Handle the final chunk (if any sentences remain)
    # This ensures no text is lost at the end of the document
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        
        # Generate comprehensive audit metadata for final chunk
        final_chunk_metadata = {
            # Core identification
            "source_file": base_filename,
            "chunk_position": "last" if chunk_index > 0 else "only",
            
            # Sentence boundaries for traceability
            "start_sentence": len(sentences) - len(current_chunk),
            "end_sentence": len(sentences) - 1,
            
            # Size metrics for validation
            "token_count": current_length,
            "character_count": len(chunk_text),
            
            # Audit trail information
            "processing_timestamp": processing_timestamp,
            "chunk_hash": hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16],
            
            # Processing context
            "overlap_with_previous": chunk_index > 0,
            "total_chunks_in_document": chunk_index + 1,
            
            # KYC/AML compliance metadata
            "compliance_flags": {
                "document_type": "financial_report" if "report" in base_filename.lower() else "unknown",
                "contains_financial_data": any(term in chunk_text.lower() for term in ["revenue", "profit", "loss", "financial", "million", "billion", "eur", "usd", "$"]),
                "contains_entity_names": any(char.isupper() for char in chunk_text if char.isalpha()),
                "requires_review": len(chunk_text) > chunk_size * 4,
                "processing_stage": "chunking",
                "audit_ready": True
            }
        }
        
        chunks.append({
            "chunk_id": chunk_index,
            "text": chunk_text,
            "metadata": final_chunk_metadata
        })
    
    # Update total_chunks_in_document for all chunks
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_chunks_in_document"] = total_chunks
        
        # Update chunk_position for better accuracy
        if total_chunks == 1:
            chunk["metadata"]["chunk_position"] = "only"
        elif chunk["chunk_id"] == 0:
            chunk["metadata"]["chunk_position"] = "first"
        elif chunk["chunk_id"] == total_chunks - 1:
            chunk["metadata"]["chunk_position"] = "last"
        else:
            chunk["metadata"]["chunk_position"] = "middle"

    return chunks


def save_chunks_to_file(chunks, file_name="output_data/chunks/chunks_output.txt"):
    """
    Persists processed chunks to disk for manual inspection and debugging.
    
    This function is essential for the development and debugging phase of the
    Graph RAG pipeline. It creates a human-readable output file that allows
    developers and compliance analysts to:
    - Verify chunk quality and boundaries
    - Inspect metadata for traceability
    - Debug chunking issues before vector embedding
    - Validate that sensitive information is properly segmented
    
    Args:
        chunks (list): List of chunk dictionaries from chunk_text() function.
                      Each chunk must contain 'chunk_id', 'text', and 'metadata' keys.
        file_name (str): Output file path where chunks will be saved.
                        Defaults to "test/chunks_output.txt" for development workflow.
                        Parent directories will be created automatically if needed.
    
    Returns:
        str: The file path where chunks were saved (same as input file_name).
             Useful for confirming successful save operation.
    
    Raises:
        PermissionError: If unable to write to the specified file location
        OSError: If directory creation fails due to system limitations
    
    Example:
        >>> chunks = chunk_text(document_text)
        >>> output_path = save_chunks_to_file(chunks, "output/analysis.txt")
        >>> print(f"Chunks saved to: {output_path}")
    
    File Format:
        The output file uses a structured format:
        ===== CHUNK 0 =====
        Metadata: {'start_sentence': 0, 'end_sentence': 5, 'length': 287}
        [chunk text content]
        
        ===== CHUNK 1 =====
        [next chunk...]
    """
    # Ensure output directory exists before writing
    # exist_ok=True prevents errors if directory already exists
    # This is crucial for automated pipeline execution
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    # Write chunks to file using UTF-8 encoding for international character support
    # This is important for compliance docs that may contain non-ASCII characters
    with open(file_name, "w", encoding="utf-8") as f:
        # Iterate through each chunk and write in structured format
        for chunk in chunks:
            # Write chunk header with clear visual separation
            # This makes manual inspection easier for analysts
            f.write(f"===== CHUNK {chunk['chunk_id']} =====\n")
            
            # Write metadata on separate line for easy parsing
            # Metadata includes sentence indices and token count for traceability
            f.write(f"Metadata: {chunk['metadata']}\n")
            
            # Write the actual chunk text content
            # This is what will be embedded and used for RAG retrieval
            f.write(chunk['text'])
            
            # Add double newline for clear visual separation between chunks
            # This improves readability during manual review
            f.write("\n\n")
    
    # Return file path for confirmation and potential chaining
    return file_name
