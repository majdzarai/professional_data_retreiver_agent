#!/usr/bin/env python3
"""
Data Loader Module for Simple RAG System

This module handles loading and reading various file formats from the input directory.
Supports text files, PDFs, and other common document formats.

Features:
- Text file loading (.txt, .md)
- PDF document extraction (requires PyPDF2)
- File metadata extraction
- Input directory statistics

Author: RAG System
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading documents from the input directory.
    
    Supports multiple file formats:
    - .txt: Plain text files
    - .md: Markdown files
    - .pdf: PDF documents (requires PyPDF2)
    """
    
    def __init__(self, input_dir: str = "input"):
        """
        Initialize the DataLoader.
        
        Args:
            input_dir (str): Path to the input directory containing documents
        """
        self.input_dir = Path(input_dir)
        self.supported_extensions = {'.txt', '.md', '.pdf'}
        
        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with input directory: {self.input_dir}")
    
    def get_supported_files(self) -> List[Path]:
        """
        Get all supported files from the input directory.
        
        Recursively searches the input directory for files with supported extensions
        (.txt, .md, .pdf) and returns their paths.
        
        Returns:
            List[Path]: List of supported file paths
        """
        files = []
        
        for file_path in self.input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                files.append(file_path)
        
        logger.info(f"Found {len(files)} supported files in {self.input_dir}")
        return files
    
    def load_text_file(self, file_path: Path) -> Optional[str]:
        """
        Load content from a text file (.txt, .md).
        
        Args:
            file_path (Path): Path to the text file
            
        Returns:
            Optional[str]: File content or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            logger.info(f"Successfully loaded text file: {file_path.name} ({len(content)} chars)")
            return content
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return None
    
    def load_pdf_file(self, file_path: Path) -> Optional[str]:
        """
        Load content from a PDF file.
        
        Args:
            file_path (Path): Path to the PDF file
            
        Returns:
            Optional[str]: Extracted text content or None if error
        """
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text_content += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
            
            logger.info(f"Successfully loaded PDF file: {file_path.name} ({len(text_content)} chars)")
            return text_content
            
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return None
    
    def load_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        Load a single file and return its metadata and content.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            Optional[Dict[str, str]]: Dictionary with file metadata and content
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Determine file type and load accordingly
        extension = file_path.suffix.lower()
        
        if extension in {'.txt', '.md'}:
            content = self.load_text_file(file_path)
        elif extension == '.pdf':
            content = self.load_pdf_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return None
        
        if content is None:
            return None
        
        # Return file metadata and content
        return {
            'filename': file_path.name,
            'filepath': str(file_path),
            'extension': extension,
            'size_chars': len(content),
            'content': content
        }
    
    def load_all_files(self) -> List[Dict[str, str]]:
        """
        Load all supported files from the input directory.
        
        Returns:
            List[Dict[str, str]]: List of loaded documents with metadata
        """
        files = self.get_supported_files()
        loaded_documents = []
        
        logger.info(f"Loading {len(files)} files...")
        
        for file_path in files:
            document = self.load_file(file_path)
            if document:
                loaded_documents.append(document)
        
        logger.info(f"Successfully loaded {len(loaded_documents)} documents")
        return loaded_documents
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about files in the input directory.
        
        Returns:
            Dict[str, int]: Statistics about file types and counts
        """
        files = self.get_supported_files()
        stats = {
            'total_files': len(files),
            'txt_files': 0,
            'md_files': 0,
            'pdf_files': 0
        }
        
        for file_path in files:
            extension = file_path.suffix.lower()
            if extension == '.txt':
                stats['txt_files'] += 1
            elif extension == '.md':
                stats['md_files'] += 1
            elif extension == '.pdf':
                stats['pdf_files'] += 1
        
        return stats


def main():
    """
    Example usage of the DataLoader.
    """
    # Initialize data loader
    loader = DataLoader("input")
    
    # Get file statistics
    stats = loader.get_stats()
    print(f"File Statistics: {stats}")
    
    # Load all documents
    documents = loader.load_all_files()
    
    # Display loaded documents
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"  Filename: {doc['filename']}")
        print(f"  Type: {doc['extension']}")
        print(f"  Size: {doc['size_chars']} characters")
        print(f"  Content preview: {doc['content'][:200]}...")


if __name__ == "__main__":
    main()