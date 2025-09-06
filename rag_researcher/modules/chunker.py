#!/usr/bin/env python3
"""
Chunking Module for Simple RAG System

This module handles splitting cleaned text into manageable chunks for processing.
Supports various chunking strategies including fixed-size, sentence-based,
and semantic chunking approaches.

Features:
- Fixed-size chunking with overlap: Splits text into chunks of specified size with configurable overlap
- Sentence-aware chunking: Respects sentence boundaries to maintain context
- Paragraph-based chunking: Uses paragraph breaks as natural chunk boundaries
- Token-based chunking: Can split based on word/token count rather than characters
- Chunk metadata and statistics: Tracks information about chunks for analysis

The chunking process is critical for RAG systems as it determines how information is segmented
and later retrieved. Good chunking preserves context while creating appropriately sized text segments.

Author: RAG System
Date: 2025
"""

import re
import logging
import math
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles text chunking operations with various strategies.
    
    This class is the core component of the chunking system. It takes cleaned text
    and splits it into smaller, manageable chunks that can be processed independently.
    The chunking strategy significantly impacts retrieval quality in RAG systems.
    
    Supports multiple chunking methods:
    - Fixed-size chunking (character or word-based): Splits text into chunks of fixed character length
    - Sentence-aware chunking: Creates chunks that preserve complete sentences
    - Paragraph-based chunking: Uses paragraphs as natural document segments
    - Sliding window with overlap: Creates overlapping chunks to preserve context across chunk boundaries
    
    The class handles metadata tracking, statistics generation, and provides multiple
    output formats for the chunked documents.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap_size: int = 200,
                 chunking_method: str = "paragraph",
                 preserve_sentences: bool = True,
                 min_chunk_size: int = 100):
        """
        Initialize the TextChunker with configuration options.
        
        The initialization sets up the chunking parameters that control how text is divided.
        These parameters significantly impact the quality of chunks and downstream retrieval:
        
        Args:
            chunk_size (int): Target size for each chunk in characters. Default 1000 chars provides
                              a good balance between context preservation and specificity.
            overlap_size (int): Number of characters that overlap between consecutive chunks.
                               Overlap helps maintain context across chunk boundaries and improves
                               retrieval of information that spans multiple chunks. Default 200.
            chunking_method (str): Strategy for splitting text:
                                  - 'fixed_size': Splits by character count (with overlap)
                                  - 'sentence': Creates chunks of complete sentences
                                  - 'paragraph': Uses paragraph breaks as natural boundaries (default)
            preserve_sentences (bool): When True, attempts to avoid splitting in the middle of sentences
                                      even in fixed_size mode by finding nearby sentence boundaries.
            min_chunk_size (int): Prevents creation of very small chunks that might lack context.
                                 Chunks smaller than this size may be merged. Default 100 chars.
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.chunking_method = chunking_method
        self.preserve_sentences = preserve_sentences
        self.min_chunk_size = min_chunk_size
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("TextChunker initialized with configuration:")
        logger.info(f"  - Chunk size: {chunk_size}")
        logger.info(f"  - Overlap size: {overlap_size}")
        logger.info(f"  - Method: {chunking_method}")
        logger.info(f"  - Preserve sentences: {preserve_sentences}")
        logger.info(f"  - Min chunk size: {min_chunk_size}")
    
    def _compile_patterns(self):
        """
        Compile regex patterns for text processing.
        
        Pre-compiles regular expressions used for identifying text boundaries.
        Compiling patterns once during initialization improves performance by avoiding
        repeated regex compilation during the chunking process.
        
        The patterns identify:
        - Sentence boundaries: Detects periods, exclamation marks, question marks followed by
          capital letters, which typically indicate the end of one sentence and start of another.
          This is crucial for sentence-preserving chunking.
          
        - Paragraph boundaries: Identifies empty lines that separate paragraphs in text.
          Used primarily in paragraph-based chunking to maintain natural document structure.
          
        - Word boundaries: Recognizes whitespace that separates words, used for word counting
          and token-based operations.
        
        These patterns are used during the chunking process to ensure chunks respect
        natural text boundaries when possible, improving readability and context preservation.
        """
        # Sentence boundaries (basic pattern)
        # This pattern looks for punctuation (.!?) followed by whitespace and a capital letter
        # The lookbehind (?<=[.!?]) matches punctuation without including it in the result
        # The lookahead (?=[A-Z]) ensures we only match when followed by capital letters
        # This helps identify sentence boundaries in English and similar languages
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])', re.MULTILINE)
        
        # Paragraph boundaries
        # Matches one or more newlines with optional whitespace between them
        # This identifies paragraph breaks in text documents
        self.paragraph_pattern = re.compile(r'\n\s*\n', re.MULTILINE)
        
        # Word boundaries
        # Matches one or more whitespace characters
        # Used for splitting text into words/tokens and counting them
        self.word_pattern = re.compile(r'\s+', re.MULTILINE)
    
    def _validate_config(self):
        """
        Validate chunker configuration.
        
        Performs sanity checks on the configuration parameters to ensure they're valid
        and logical. Raises appropriate exceptions for invalid configurations and logs
        warnings for potentially problematic settings.
        """
        # Ensure chunk size is positive - negative or zero chunk size would be meaningless
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        # Ensure overlap size is not negative - negative overlap doesn't make sense
        if self.overlap_size < 0:
            raise ValueError("Overlap size cannot be negative")
        
        # Warn if overlap is larger than or equal to chunk size
        # This would create redundant chunks with complete overlap
        if self.overlap_size >= self.chunk_size:
            logger.warning(f"Overlap size ({self.overlap_size}) >= chunk size ({self.chunk_size})")
        
        # Validate that the chunking method is one of the supported options
        # This prevents runtime errors when trying to use an unsupported method
        if self.chunking_method not in ['fixed_size', 'sentence', 'paragraph']:
            raise ValueError(f"Invalid chunking method: {self.chunking_method}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        This method uses the pre-compiled sentence pattern to identify sentence boundaries
        and split the text accordingly. It's used as a building block for sentence-aware
        chunking strategies and for preserving sentence boundaries in fixed-size chunking.
        
        The method also performs basic cleaning and filtering to remove very short
        sentences that might be noise or incomplete fragments.
        
        Args:
            text (str): Input text to be split into sentences
            
        Returns:
            List[str]: List of cleaned sentences extracted from the text
        """
        if not text:
            return []
        
        # Split the text at sentence boundaries using the pre-compiled regex pattern
        # This creates a list of text segments that should each be a complete sentence
        sentences = self.sentence_pattern.split(text)
        
        # Clean up sentences and filter out very short ones
        cleaned_sentences = []
        for sentence in sentences:
            # Remove leading/trailing whitespace
            sentence = sentence.strip()
            
            # Only keep sentences that have content and are at least 10 characters
            # This helps filter out very short fragments that might not be meaningful
            if sentence and len(sentence) >= 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        This method identifies paragraph boundaries (typically blank lines) in the text
        and splits it accordingly. Paragraphs are natural semantic units in most documents
        and often make good chunk boundaries for RAG systems.
        
        The method also performs cleaning and filtering to ensure paragraphs meet the
        minimum size requirements, avoiding overly small chunks.
        
        Args:
            text (str): Input text to be split into paragraphs
            
        Returns:
            List[str]: List of cleaned paragraphs extracted from the text
        """
        if not text:
            return []
        
        # Split the text at paragraph boundaries (blank lines) using the pre-compiled regex
        # This creates a list of text segments that should each be a complete paragraph
        paragraphs = self.paragraph_pattern.split(text)
        
        # Clean up paragraphs and filter out ones that are too small
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            # Remove leading/trailing whitespace
            paragraph = paragraph.strip()
            
            # Only keep paragraphs that have content and meet the minimum size requirement
            # This prevents creating very small chunks that might lack sufficient context
            if paragraph and len(paragraph) >= self.min_chunk_size:
                cleaned_paragraphs.append(paragraph)
        
        return cleaned_paragraphs
    
    def fixed_size_chunking(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        This method creates chunks of approximately equal size (specified by chunk_size),
        with optional overlap between consecutive chunks. Fixed-size chunking ensures
        consistent chunk sizes but may split text at arbitrary points.
        
        When preserve_sentences is enabled, the method attempts to find sentence boundaries
        near the chunk boundaries to avoid splitting in the middle of sentences, which
        improves readability and context preservation.
        
        Args:
            text (str): Input text to be split into fixed-size chunks
            
        Returns:
            List[str]: List of text chunks of approximately equal size
        """
        # Return empty list for empty input
        if not text:
            return []
        
        chunks = []  # List to store the resulting chunks
        text_length = len(text)  # Total length of the input text
        start = 0  # Starting position for the current chunk
        
        # Continue chunking until we've processed the entire text
        while start < text_length:
            # Calculate the end position for the current chunk
            # This is either the start position plus chunk_size or the end of the text
            end = min(start + self.chunk_size, text_length)
            
            # Extract the chunk from the text using the calculated boundaries
            chunk = text[start:end]
            
            # If sentence preservation is enabled and we're not at the end of the text,
            # try to adjust the chunk boundary to end at a sentence boundary
            if self.preserve_sentences and end < text_length:
                # Look for sentence boundaries within the last 20% of the chunk
                # This gives us some flexibility in finding a good boundary
                search_start = max(start + int(self.chunk_size * 0.8), start)
                
                # Look a bit ahead of the current end position to find sentence boundaries
                # The +100 provides a buffer to find the next sentence boundary
                search_text = text[search_start:end + 100]  # Look a bit ahead
                
                # Find the last sentence boundary in the search text
                # Initialize variable to store the last sentence boundary match found
                sentence_match = None
                
                # Iterate through all sentence boundaries in the search text
                # and keep track of the last one found
                for match in self.sentence_pattern.finditer(search_text):
                    sentence_match = match
                
                # If we found a sentence boundary, adjust the chunk end position
                if sentence_match:
                    # Adjust end to sentence boundary
                    new_end = search_start + sentence_match.end()
                    if new_end > start + self.min_chunk_size:
                        end = min(new_end, text_length)
                        chunk = text[start:end]
            
            # Add chunk if it meets minimum size
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append(chunk.strip())
            
            # Calculate next start position with overlap
            if end >= text_length:
                break
            
            # Move to the next chunk position, accounting for the configured overlap
            # This creates overlapping chunks which helps maintain context between chunks
            start = max(end - self.overlap_size, start + 1)
        
        return chunks
    
    def sentence_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences. This method preserves sentence boundaries
        and creates chunks that contain complete sentences, with optional overlap between chunks.
        
        Args:
            text (str): Input text to be chunked into sentences
            
        Returns:
            List[str]: List of sentence-based chunks that respect the configured size limits
        """
        # First, split the input text into individual sentences
        sentences = self.split_into_sentences(text)
        
        # Return empty list if no sentences were found
        if not sentences:
            return []
        
        chunks = []  # List to store the final chunks
        current_chunk = ""  # Buffer for building the current chunk
        current_size = 0  # Track the size of the current chunk
        
        # Process each sentence one by one
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed the maximum chunk size
            # and if we already have content in the current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save the completed chunk to our results list
                chunks.append(current_chunk.strip())
                
                # Handle overlap between chunks if configured
                if self.overlap_size > 0:
                    # Create overlap by including the last portion of the previous chunk
                    # This helps maintain context between chunks
                    overlap_text = current_chunk[-self.overlap_size:]
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk)
                else:
                    # No overlap, just start a new chunk with this sentence
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                # The sentence fits in the current chunk, so add it
                if current_chunk:
                    # Add space between sentences
                    current_chunk += " " + sentence
                else:
                    # This is the first sentence in this chunk
                    current_chunk = sentence
                    
                # Update the current chunk size
                current_size = len(current_chunk)
        
        # Don't forget to add the final chunk if it exists and meets minimum size requirements
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Return all the sentence-based chunks
        return chunks
    
    def paragraph_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraphs. This method attempts to keep paragraphs
        together when possible, but will split large paragraphs if necessary.
        
        Args:
            text (str): Input text to be chunked
            
        Returns:
            List[str]: List of paragraph-based chunks that respect the configured size limits
        """
        # First, split the input text into individual paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        # Return empty list if no paragraphs were found
        if not paragraphs:
            return []
        
        chunks = []  # List to store the final chunks
        current_chunk = ""  # Buffer for building the current chunk
        current_size = 0  # Track the size of the current chunk
        
        # Process each paragraph one by one
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # Handle paragraphs that are larger than the maximum chunk size
            # These need to be split into smaller pieces
            if paragraph_size > self.chunk_size:
                # First save any accumulated chunk content if it exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""  # Reset the chunk buffer
                    current_size = 0  # Reset the size counter
                
                # Use fixed-size chunking to break down the large paragraph
                # This ensures no chunk exceeds the maximum size
                para_chunks = self.fixed_size_chunking(paragraph)
                chunks.extend(para_chunks)  # Add all sub-chunks to our results
                continue  # Move to the next paragraph
            
            # Check if adding this paragraph would exceed the maximum chunk size
            # and if we already have content in the current chunk
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                # Save the completed chunk to our results list
                chunks.append(current_chunk.strip())
                
                # Start a new chunk with this paragraph
                current_chunk = paragraph
                current_size = paragraph_size
            else:
                # The paragraph fits in the current chunk, so add it
                if current_chunk:
                    # Add paragraph separator (double newline) between paragraphs
                    current_chunk += "\n\n" + paragraph
                else:
                    # This is the first paragraph in this chunk
                    current_chunk = paragraph
                    
                # Update the current chunk size
                current_size = len(current_chunk)
        
        # Don't forget to add the final chunk if it exists and meets minimum size requirements
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Return all the paragraph-based chunks
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using the configured chunking method. This is the main entry point
        for text chunking that dispatches to the appropriate specialized chunking method.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[str]: List of text chunks created according to the configured method
        """
        # Return empty list for invalid input
        if not text or not isinstance(text, str):
            return []
        
        # Dispatch to the appropriate chunking method based on configuration
        if self.chunking_method == "sentence":
            # Sentence-based chunking preserves sentence boundaries
            chunks = self.sentence_chunking(text)
        elif self.chunking_method == "paragraph":
            # Paragraph-based chunking tries to keep paragraphs together
            chunks = self.paragraph_chunking(text)
        else:  # fixed_size
            # Fixed-size chunking creates chunks of approximately equal size
            chunks = self.fixed_size_chunking(text)
        
        # Log chunking results
        logger.info(f"Chunked text into {len(chunks)} chunks using {self.chunking_method} method")
        
        # Return the resulting chunks
        return chunks
    
    def chunk_document(self, document: Dict[str, str]) -> Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]:
        """
        Process a document by chunking its content and adding metadata to each chunk.
        This method creates a structured representation of the document with its chunks
        and relevant metadata for both the document and individual chunks.
        
        Args:
            document (Dict[str, str]): Document with 'content' key
            
        Returns:
            Dict: Document with chunks and metadata
        """
        if 'content' not in document:
            logger.warning("Document missing 'content' key")
            return document
        
        content = document['content']
        chunks = self.chunk_text(content)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            # Create a structured chunk object with metadata including size and preview
            chunk_obj = {
                'chunk_id': i,
                'content': chunk,
                'size_chars': len(chunk),
                'size_words': len(chunk.split()),
                'start_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            chunk_objects.append(chunk_obj)
        
        # Create a comprehensive result document with both document and chunking metadata
        result = {
            # Document metadata
            'filename': document.get('filename', 'unknown'),  # Original filename or default
            'filepath': document.get('filepath', ''),  # Original file path
            'original_size_chars': len(content),  # Size of the original document in characters
            
            # Chunking metadata
            'total_chunks': len(chunks),  # Number of chunks created
            'chunking_method': self.chunking_method,  # Method used for chunking
            'chunk_size': self.chunk_size,  # Configured maximum chunk size
            'overlap_size': self.overlap_size,  # Configured overlap between chunks
            
            # The actual chunks with their metadata
            'chunks': chunk_objects
        }
        
        # Calculate and add statistical information about the chunks
        if chunks:
            # Get the size of each chunk for statistical analysis
            chunk_sizes = [len(chunk) for chunk in chunks]
            
            # Calculate average, minimum, and maximum chunk sizes
            result['avg_chunk_size'] = sum(chunk_sizes) / len(chunk_sizes)
            result['min_chunk_size'] = min(chunk_sizes)
            result['max_chunk_size'] = max(chunk_sizes)
        
        # Log a summary of the chunking operation for monitoring
        logger.info(f"Chunked document '{document.get('filename', 'unknown')}': "
                   f"{len(content)} chars → {len(chunks)} chunks "
                   f"(avg: {result.get('avg_chunk_size', 0):.0f} chars/chunk)")
        
        return result
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]:  
        """
        Process and chunk multiple documents in batch. This method iterates through a list of documents,
        chunks each one individually, and aggregates statistics about the overall chunking operation.
        
        Args:
            documents (List[Dict[str, str]]): List of document dictionaries to chunk
            
        Returns:
            List[Dict]: List of chunked documents with metadata for each document
        """
        # Initialize result list and statistics counters
        chunked_documents = []  # Will hold all processed documents
        total_chunks = 0  # Counter for total chunks created
        total_original_size = 0  # Counter for total input size in characters
        
        # Log the start of the batch chunking process
        logger.info(f"Chunking {len(documents)} documents...")
        
        # Process each document individually
        for document in documents:
            # Chunk the current document
            chunked_doc = self.chunk_document(document)
            
            # Add the chunked document to our results
            chunked_documents.append(chunked_doc)
            
            # Aggregate statistics from this document
            total_chunks += chunked_doc.get('total_chunks', 0)
            total_original_size += chunked_doc.get('original_size_chars', 0)
        
        # Calculate the average number of chunks per document
        avg_chunks_per_doc = total_chunks / len(documents) if documents else 0
        
        # Log summary statistics about the chunking operation
        logger.info(f"Chunking completed:")
        logger.info(f"  - Total documents: {len(documents)}")
        logger.info(f"  - Total chunks: {total_chunks}")
        logger.info(f"  - Average chunks per document: {avg_chunks_per_doc:.1f}")
        logger.info(f"  - Total original size: {total_original_size:,} chars")
        
        # Return the list of chunked documents
        return chunked_documents
    
    def save_chunks(self, chunked_documents: List[Dict], output_dir: str = "output"):
        """
        Save chunked documents to output files in both JSON and human-readable text formats.
        The JSON format preserves all metadata and is suitable for programmatic use,
        while the text format is optimized for human readability and inspection.
        
        Args:
            chunked_documents (List[Dict]): List of chunked document objects with metadata
            output_dir (str): Directory path where output files will be saved
        """
        # Create the output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each chunked document
        for doc in chunked_documents:
            # Extract the original filename and create a base name for output files
            filename = doc.get('filename', 'unknown')
            base_name = Path(filename).stem  # Get filename without extension
            
            # Save the complete document with all chunks and metadata as JSON
            # This format preserves all information and is suitable for programmatic use
            json_file = output_path / f"{base_name}_chunks.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)  # Pretty-print with indentation
            
            # Save chunks in a human-readable text format for easy inspection
            txt_file = output_path / f"{base_name}_chunks.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                # Write document header information
                f.write(f"Document: {filename}\n")
                f.write(f"Total chunks: {doc.get('total_chunks', 0)}\n")
                f.write(f"Chunking method: {doc.get('chunking_method', 'unknown')}\n")
                f.write("=" * 50 + "\n\n")  # Separator line
                
                # Write each chunk with its metadata and content
                for chunk in doc.get('chunks', []):
                    # Write chunk header with ID and size
                    f.write(f"Chunk {chunk['chunk_id']} ({chunk['size_chars']} chars):\n")
                    f.write("-" * 30 + "\n")  # Separator line
                    # Write the actual chunk content
                    f.write(chunk['content'])
                    f.write("\n\n")  # Add spacing between chunks
        
        logger.info(f"Saved {len(chunked_documents)} chunked documents to {output_path}")
    
    def get_chunking_stats(self, chunked_documents: List[Dict]) -> Dict[str, float]:
        """
        Calculate comprehensive statistics about the chunking process across all documents.
        This method aggregates data from all chunked documents to provide insights into
        the chunking performance and characteristics.
        
        Args:
            chunked_documents (List[Dict]): List of chunked document objects with metadata
            
        Returns:
            Dict[str, float]: Dictionary containing various statistics about the chunking process
        """
        # Return empty dictionary if no documents were provided
        if not chunked_documents:
            return {}
        
        # Calculate basic document-level statistics
        total_docs = len(chunked_documents)  # Total number of documents processed
        total_chunks = sum(doc.get('total_chunks', 0) for doc in chunked_documents)  # Total chunks created
        total_original_size = sum(doc.get('original_size_chars', 0) for doc in chunked_documents)  # Total input size
        
        # Collect size information for all individual chunks across all documents
        # This allows us to calculate statistics about chunk sizes
        all_chunk_sizes = []
        for doc in chunked_documents:
            for chunk in doc.get('chunks', []):
                all_chunk_sizes.append(chunk.get('size_chars', 0))
        
        # Create a comprehensive statistics dictionary with detailed information
        return {
            'total_documents': total_docs,  # Number of documents processed
            'total_chunks': total_chunks,  # Total number of chunks created
            'total_original_chars': total_original_size,  # Total size of input documents in characters
            'avg_chunks_per_doc': total_chunks / total_docs if total_docs > 0 else 0,  # Average chunks per document
            'avg_chunk_size': sum(all_chunk_sizes) / len(all_chunk_sizes) if all_chunk_sizes else 0,  # Average chunk size
            'min_chunk_size': min(all_chunk_sizes) if all_chunk_sizes else 0,  # Size of smallest chunk
            'max_chunk_size': max(all_chunk_sizes) if all_chunk_sizes else 0,  # Size of largest chunk
            'chunking_method': self.chunking_method,  # Method used for chunking
            'configured_chunk_size': self.chunk_size,  # Configured maximum chunk size
            'configured_overlap': self.overlap_size  # Configured overlap between chunks
        }


def main():
    """
    Example usage of the TextChunker.
    """
    # Sample text
    sample_text = """
    This is the first paragraph of our sample document. It contains multiple sentences that will be used to test the chunking functionality. The chunker should be able to handle this text appropriately.
    
    This is the second paragraph. It's a bit shorter than the first one. But it still contains enough content to be meaningful for testing purposes.
    
    The third paragraph is here to provide more content. We want to see how the chunker handles multiple paragraphs and whether it preserves the structure appropriately. This should give us good insights into the chunking behavior.
    
    Finally, this is the last paragraph of our sample document. It concludes our test content and should be processed along with all the previous paragraphs.
    """
    
    # Test different chunking methods
    methods = ['fixed_size', 'sentence', 'paragraph']
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} chunking ===")
        
        chunker = TextChunker(
            chunk_size=200,
            overlap_size=50,
            chunking_method=method,
            preserve_sentences=True
        )
        
        chunks = chunker.chunk_text(sample_text)
        
        print(f"Generated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} chars):")
            print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)


if __name__ == "__main__":
    main()