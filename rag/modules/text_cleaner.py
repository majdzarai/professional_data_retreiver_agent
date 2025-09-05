#!/usr/bin/env python3
"""
Text Cleaner Module for Simple RAG System

This module handles text preprocessing and cleaning operations to prepare
raw text data for chunking and further processing.

Features:
- Remove unwanted characters and formatting
- Normalize whitespace and line breaks
- Handle special characters and encoding issues
- Remove or preserve specific patterns
- Text normalization and standardization

Author: RAG System
Date: 2025
"""

import re
import logging
import unicodedata
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Handles text cleaning and preprocessing operations.
    
    Provides various cleaning methods to prepare text for chunking:
    - Basic cleaning (whitespace, special chars)
    - Advanced cleaning (encoding, normalization)
    - Custom pattern removal
    - Text standardization
    """
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_special_chars: bool = False,
                 preserve_line_breaks: bool = True):
        """
        Initialize the TextCleaner with configuration options.
        
        Args:
            remove_extra_whitespace (bool): Remove extra spaces and tabs
            normalize_unicode (bool): Normalize unicode characters
            remove_special_chars (bool): Remove special characters
            preserve_line_breaks (bool): Keep line breaks in text
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.preserve_line_breaks = preserve_line_breaks
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("TextCleaner initialized with configuration:")
        logger.info(f"  - Remove extra whitespace: {remove_extra_whitespace}")
        logger.info(f"  - Normalize unicode: {normalize_unicode}")
        logger.info(f"  - Remove special chars: {remove_special_chars}")
        logger.info(f"  - Preserve line breaks: {preserve_line_breaks}")
    
    def _compile_patterns(self):
        """
        Compile regex patterns for text cleaning.
        
        Pre-compiles regular expression patterns for efficient text processing.
        These patterns are used throughout the cleaning process for various operations.
        """
        # Multiple consecutive whitespace (spaces, tabs)
        self.whitespace_pattern = re.compile(r'[ \t]+', re.MULTILINE)
        
        # Multiple consecutive line breaks
        self.linebreak_pattern = re.compile(r'\n{3,}', re.MULTILINE)
        
        # Special characters (keep basic punctuation)
        self.special_chars_pattern = re.compile(r'[^\w\s.,!?;:()\[\]{}"\'-]', re.UNICODE)
        
        # Email addresses
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # URLs
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Phone numbers (basic pattern)
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b')
        
        # Extra punctuation
        self.extra_punct_pattern = re.compile(r'[.,!?;:]{2,}')
    
    def basic_clean(self, text: str) -> str:
        """
        Perform basic text cleaning operations.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        cleaned_text = text
        
        # Remove or normalize unicode characters
        if self.normalize_unicode:
            cleaned_text = unicodedata.normalize('NFKD', cleaned_text)
            # Remove non-ASCII characters that can't be encoded
            cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text)
        
        # Handle line breaks
        if self.preserve_line_breaks:
            # Reduce multiple line breaks to maximum of 2
            cleaned_text = self.linebreak_pattern.sub('\n\n', cleaned_text)
        else:
            # Replace all line breaks with spaces
            cleaned_text = cleaned_text.replace('\n', ' ')
            cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text)
        
        # Remove special characters if requested
        if self.remove_special_chars:
            cleaned_text = self.special_chars_pattern.sub(' ', cleaned_text)
            cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text)
        
        # Clean up extra punctuation
        cleaned_text = self.extra_punct_pattern.sub(lambda m: m.group(0)[0], cleaned_text)
        
        # Strip leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def remove_patterns(self, text: str, patterns: List[str]) -> str:
        """
        Remove custom patterns from text.
        
        Args:
            text (str): Input text
            patterns (List[str]): List of regex patterns to remove
            
        Returns:
            str: Text with patterns removed
        """
        cleaned_text = text
        
        for pattern in patterns:
            try:
                compiled_pattern = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                cleaned_text = compiled_pattern.sub(' ', cleaned_text)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        # Clean up extra whitespace after pattern removal
        if self.remove_extra_whitespace:
            cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def remove_sensitive_info(self, text: str) -> str:
        """
        Remove potentially sensitive information from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with sensitive info removed/masked
        """
        cleaned_text = text
        
        # Remove email addresses
        cleaned_text = self.email_pattern.sub('[EMAIL]', cleaned_text)
        
        # Remove URLs
        cleaned_text = self.url_pattern.sub('[URL]', cleaned_text)
        
        # Remove phone numbers
        cleaned_text = self.phone_pattern.sub('[PHONE]', cleaned_text)
        
        return cleaned_text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent processing.
        
        Performs standardization of various text elements:
        - Converts smart/curly quotes to straight quotes
        - Standardizes apostrophes and single quotes
        - Normalizes various dash types to simple hyphens
        - Converts Unicode ellipsis to ASCII representation
        - Fixes spacing around punctuation marks
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text with standardized characters and spacing
        """
        if not text:
            return ""
        
        normalized_text = text
        
        # Normalize quotes
        normalized_text = re.sub(r'[“”]', '"', normalized_text)
        normalized_text = re.sub(r'[‘’]', "'", normalized_text)
        
        # Normalize dashes
        normalized_text = re.sub(r'—', '-', normalized_text)
        normalized_text = re.sub(r'–', '-', normalized_text)
        
        # Normalize ellipsis
        normalized_text = re.sub(r'…', '...', normalized_text)
        
        # Normalize spacing around punctuation
        normalized_text = re.sub(r'\s+([.,!?;:])', r'\1', normalized_text)
        normalized_text = re.sub(r'([.,!?;:])\s+', r'\1 ', normalized_text)
        
        return normalized_text.strip()
    
    def clean_document(self, text: str, 
                      custom_patterns: Optional[List[str]] = None,
                      remove_sensitive: bool = False) -> str:
        """
        Clean a single document with all available cleaning methods.
        
        Args:
            text (str): Input text to clean
            custom_patterns (List[str], optional): Custom regex patterns to remove
            remove_sensitive (bool): Whether to remove sensitive information
            
        Returns:
            str: Fully cleaned text
        """
        if not text:
            return ""
        
        # Start with basic cleaning
        cleaned_text = self.basic_clean(text)
        
        # Apply text normalization
        cleaned_text = self.normalize_text(cleaned_text)
        
        # Remove custom patterns if provided
        if custom_patterns:
            cleaned_text = self.remove_patterns(cleaned_text, custom_patterns)
        
        # Remove sensitive information if requested
        if remove_sensitive:
            cleaned_text = self.remove_sensitive_info(cleaned_text)
        
        return cleaned_text
    
    def clean_documents(self, documents: List[Dict[str, str]], 
                       custom_patterns: Optional[List[str]] = None,
                       remove_sensitive: bool = False) -> List[Dict[str, str]]:
        """
        Clean multiple documents.
        
        Args:
            documents (List[Dict]): List of document dictionaries with 'content' key
            custom_patterns (List[str], optional): Custom regex patterns to remove
            remove_sensitive (bool): Whether to remove sensitive information
            
        Returns:
            List[Dict]: List of cleaned documents
        """
        cleaned_documents = []
        
        for doc in documents:
            if 'content' in doc:
                cleaned_doc = doc.copy()
                cleaned_doc['content'] = self.clean_document(
                    doc['content'], 
                    custom_patterns, 
                    remove_sensitive
                )
                cleaned_documents.append(cleaned_doc)
            else:
                logger.warning("Document missing 'content' key, skipping")
        
        return cleaned_documents
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, int]:
        """
        Get statistics about the cleaning process.
        
        Args:
            original_text (str): Original text before cleaning
            cleaned_text (str): Text after cleaning
            
        Returns:
            Dict[str, int]: Statistics about the cleaning process
        """
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'characters_removed': len(original_text) - len(cleaned_text),
            'original_lines': original_text.count('\n') + 1,
            'cleaned_lines': cleaned_text.count('\n') + 1,
            'original_words': len(original_text.split()),
            'cleaned_words': len(cleaned_text.split())
        }


def main():
    """
    Example usage of the TextCleaner.
    """
    # Example text with various issues
    sample_text = """
    This is a sample text with    extra   spaces.
    
    
    
    It has multiple line breaks and special characters: @#$%^&*
    
    Contact us at: john.doe@example.com or visit https://example.com
    
    Phone: +1-555-123-4567
    
    "Smart quotes" and 'apostrophes' need normalization.
    """
    
    # Initialize cleaner
    cleaner = TextCleaner(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        remove_special_chars=True,
        preserve_line_breaks=True
    )
    
    # Clean the text
    cleaned = cleaner.clean_document(
        sample_text, 
        remove_sensitive=True
    )
    
    # Get statistics
    stats = cleaner.get_cleaning_stats(sample_text, cleaned)
    
    print("Original text:")
    print(repr(sample_text))
    print("\nCleaned text:")
    print(repr(cleaned))
    print("\nCleaning statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()