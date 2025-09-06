#!/usr/bin/env python3
"""
Advanced PDF and Text Processor for Knowledge Graph Pipeline

This module provides comprehensive PDF text extraction and text cleaning capabilities
optimized for Large Language Model (LLM) processing and knowledge graph construction.

Features:
- Multi-method PDF text extraction (PyPDF2, PyMuPDF)
- OCR support for scanned PDFs (Tesseract)
- Advanced text cleaning for LLM optimization
- Batch processing of PDF and text files
- Robust error handling and progress reporting

Dependencies:
- Required: PyPDF2 (basic PDF text extraction)
- Optional: PyMuPDF (enhanced PDF handling, better for tables)
- Optional: pytesseract + PIL (OCR for scanned PDFs)

Author: Knowledge Graph Pipeline
Version: 2.0
"""

import os
import re
from pathlib import Path

# ============================================================================
# DEPENDENCY IMPORTS AND AVAILABILITY CHECKS
# ============================================================================

# Core PDF processing library (required)
try:
    import PyPDF2  # Standard library for PDF text extraction
except ImportError:
    print("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
    exit(1)

# Enhanced PDF processing library (optional but recommended)
try:
    import fitz  # PyMuPDF - better handling of complex layouts, tables, and images
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️  PyMuPDF not available. Install with 'pip install PyMuPDF' for enhanced PDF support")

# OCR libraries for scanned PDF processing (optional)
try:
    import pytesseract  # Tesseract OCR engine wrapper
    from PIL import Image  # Python Imaging Library for image processing
    import io  # Input/output operations for image data
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR libraries not available. Install with 'pip install pytesseract pillow' for scanned PDF support")

# ============================================================================
# CORE PDF TEXT EXTRACTION FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """
    Master PDF text extraction function using a multi-method approach.
    
    This function attempts multiple extraction methods in order of effectiveness:
    1. PyMuPDF (best for complex layouts, tables, and formatted documents)
    2. PyPDF2 (reliable fallback for standard PDFs)
    3. OCR with Tesseract (for scanned PDFs when text extraction yields minimal results)
    
    The function automatically selects the best method based on:
    - Available libraries
    - Quality of extracted text
    - Document type (regular vs scanned)
    
    Args:
        pdf_path (str): Absolute or relative path to the PDF file to process
        
    Returns:
        str: Extracted and formatted text content with page separators
             Returns empty string if all extraction methods fail
             
    Example:
        >>> text = extract_text_from_pdf("document.pdf")
        >>> print(f"Extracted {len(text)} characters")
    """
    extracted_text = ""
    
    # Method 1: PyMuPDF extraction (preferred for complex documents)
    if PYMUPDF_AVAILABLE:
        try:
            print(f"  → Attempting PyMuPDF extraction (enhanced method)...")
            extracted_text = extract_with_pymupdf(pdf_path)
            if extracted_text.strip():  # Success - return immediately if we got meaningful content
                print(f"  ✅ PyMuPDF extraction successful ({len(extracted_text)} characters)")
                return extracted_text
        except Exception as e:
            print(f"  ❌ PyMuPDF extraction failed: {e}")
    
    # Method 2: PyPDF2 extraction (reliable fallback)
    try:
        print(f"  → Attempting PyPDF2 extraction (standard method)...")
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Process each page individually for better error handling
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only include pages with actual content
                        extracted_text += f"\n=== Page {page_num + 1} ===\n"
                        extracted_text += page_text + "\n"
                except Exception as page_error:
                    print(f"  ⚠️  Warning: Failed to extract page {page_num + 1}: {page_error}")
                    continue
                    
        if extracted_text.strip():
            print(f"  ✅ PyPDF2 extraction successful ({len(extracted_text)} characters)")
                    
    except Exception as e:
        print(f"  ❌ PyPDF2 extraction failed: {e}")
        return ""
    
    # Method 3: OCR extraction (last resort for scanned documents)
    if len(extracted_text.strip()) < 100 and OCR_AVAILABLE and PYMUPDF_AVAILABLE:
        print(f"  → Text yield low ({len(extracted_text)} chars), attempting OCR for scanned content...")
        ocr_text = extract_with_ocr(pdf_path)
        if len(ocr_text.strip()) > len(extracted_text.strip()):
            print(f"  ✅ OCR extraction yielded better results ({len(ocr_text)} characters)")
            extracted_text = ocr_text
        
    return extracted_text

def extract_with_pymupdf(pdf_path):
    """
    Enhanced PDF text extraction using PyMuPDF (fitz) library.
    
    PyMuPDF provides superior text extraction capabilities compared to PyPDF2,
    especially for documents with:
    - Complex layouts and formatting
    - Tables and structured data
    - Mixed text and image content
    - Non-standard fonts and encodings
    
    Args:
        pdf_path (str): Path to the PDF file to process
        
    Returns:
        str: Extracted text with page separators, empty string if extraction fails
        
    Raises:
        Exception: Catches and logs any PyMuPDF-specific errors
    """
    extracted_text = ""
    document = None
    
    try:
        # Open PDF document using PyMuPDF
        document = fitz.open(pdf_path)
        
        # Process each page sequentially
        for page_num in range(len(document)):
            try:
                # Load individual page
                page = document.load_page(page_num)
                
                # Extract text with layout preservation
                # "text" mode preserves spacing and basic formatting
                page_text = page.get_text("text")
                
                # Only include pages with meaningful content
                if page_text.strip():
                    extracted_text += f"\n=== Page {page_num + 1} ===\n"
                    extracted_text += page_text + "\n"
                    
            except Exception as page_error:
                print(f"    ⚠️  Warning: PyMuPDF failed on page {page_num + 1}: {page_error}")
                continue
        
    except Exception as e:
        print(f"    ❌ PyMuPDF extraction error: {e}")
        return ""
    
    finally:
        # Ensure document is properly closed to free memory
        if document:
            document.close()
    
    return extracted_text

def extract_with_ocr(pdf_path):
    """
    OCR-based text extraction for scanned PDFs using Tesseract.
    
    This function is used as a last resort when standard PDF text extraction
    methods yield minimal results, typically for:
    - Scanned documents (images of text)
    - PDFs created from photocopies
    - Documents with embedded images containing text
    - PDFs with non-extractable fonts
    
    The OCR process:
    1. Converts each PDF page to a high-resolution image
    2. Applies Tesseract OCR with optimized settings
    3. Combines results with page markers
    
    Args:
        pdf_path (str): Path to the PDF file to process with OCR
        
    Returns:
        str: OCR-extracted text with (OCR) page markers, empty string if OCR fails
        
    Note:
        OCR accuracy depends on:
        - Image quality and resolution
        - Text clarity and font type
        - Language settings (currently optimized for English)
        - Document layout complexity
    """
    ocr_text = ""
    document = None
    
    try:
        # Open PDF document for image extraction
        document = fitz.open(pdf_path)
        
        # Process each page through OCR pipeline
        for page_num in range(len(document)):
            try:
                # Load page for image conversion
                page = document.load_page(page_num)
                
                # Convert page to high-resolution image (pixmap)
                # Higher resolution improves OCR accuracy
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better OCR
                img_data = pixmap.tobytes("png")
                
                # Convert to PIL Image for Tesseract processing
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Perform OCR with optimized Tesseract settings
                # PSM 6: Uniform block of text (good for documents)
                page_text = pytesseract.image_to_string(
                    pil_image, 
                    config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:()[]{}"\'-/'
                )
                
                # Include page only if OCR found meaningful text
                if page_text.strip():
                    ocr_text += f"\n=== Page {page_num + 1} (OCR) ===\n"
                    ocr_text += page_text + "\n"
                    
            except Exception as page_error:
                print(f"    ⚠️  Warning: OCR failed on page {page_num + 1}: {page_error}")
                continue
        
    except Exception as e:
        print(f"    ❌ OCR extraction error: {e}")
        return ""
    
    finally:
        # Clean up document resources
        if document:
            document.close()
    
    return ocr_text

def clean_text_for_llm(text):
    """
    Advanced text cleaning pipeline to optimize PDF-extracted content for LLM processing.
    
    This function removes common PDF extraction artifacts and formatting issues that can
    interfere with LLM understanding and chunking algorithms. The cleaning process includes:
    
    1. PDF artifact removal (headers, footers, page numbers)
    2. Formatting normalization (spacing, line breaks)
    3. OCR error correction (common character substitutions)
    4. Structure preservation (maintaining meaningful paragraphs)
    5. Metadata filtering (removing system-generated content)
    
    Args:
        text (str): Raw text extracted from PDF (via PyMuPDF, PyPDF2, or OCR)
        
    Returns:
        str: Cleaned and normalized text optimized for:
             - LLM chunking algorithms
             - Entity extraction
             - Relationship extraction
             - Semantic analysis
             
    Note:
        The cleaning is conservative to preserve important content while removing noise.
        Some domain-specific formatting may require additional custom cleaning rules.
    """
    # Step 1: Remove common PDF artifacts and control characters
    text = re.sub(r'\x0c', '', text)         # Remove form feed characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Remove control characters
    
    # Step 2: Remove page markers and headers
    text = re.sub(r'=== Page \d+ ===\n?', '', text)  # Remove page separators
    text = re.sub(r'=== Page \d+ \(OCR\) ===\n?', '', text)  # Remove OCR page separators
    
    # Step 3: Remove common headers and footers patterns
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip common header/footer patterns
        if (re.match(r'^Page \d+', line, re.IGNORECASE) or
            re.match(r'^\d+$', line) or  # Page numbers
            re.match(r'^[\d\-/]+$', line) or  # Date patterns
            len(line) < 3 or  # Very short lines (likely artifacts)
            line.lower() in ['confidential', 'proprietary', 'draft', 'internal use only'] or
            re.match(r'^[A-Z\s]{10,}$', line) and len(line.split()) < 5):  # All caps headers
            continue
            
        # Clean the line
        # Remove excessive spaces but preserve single spaces
        line = re.sub(r'\s+', ' ', line)
        
        # Remove bullet points and list markers for cleaner text
        line = re.sub(r'^[•·▪▫◦‣⁃]\s*', '', line)
        line = re.sub(r'^[\d]+\.\s*', '', line)  # Remove numbered list markers
        line = re.sub(r'^[a-zA-Z]\.\s*', '', line)  # Remove lettered list markers
        
        cleaned_lines.append(line)
    
    # Step 4: Rejoin and remove excessive whitespace
    text = '\n'.join(cleaned_lines)
    
    # Step 5: Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Step 6: Fix common OCR errors
    if '(OCR)' in text:
        text = re.sub(r'\bl\b', 'I', text)  # Common l/I confusion
        text = re.sub(r'\b0\b', 'O', text)  # Common 0/O confusion in words
    
    # Step 7: Remove metadata headers if present
    text = re.sub(r'^PDF Info:.*?EXTRACTED TEXT:\n={50}\n', '', text, flags=re.DOTALL)
    
    return text.strip()

def clean_text(text):
    """
    Legacy clean_text function - kept for backward compatibility.
    Use clean_text_for_llm for better LLM processing.
    """
    return clean_text_for_llm(text)

def process_text_file(txt_path, output_folder="output_data/cleaned_data"):
    """
    Process and clean existing text files for optimal LLM consumption.
    
    This function handles pre-existing text files (e.g., .txt files from previous
    extractions or manual conversions) by applying the same cleaning pipeline
    used for PDF-extracted content. This ensures consistency across all text
    sources in the knowledge graph pipeline.
    
    Processing steps:
    1. Read text file with UTF-8 encoding
    2. Apply LLM-optimized cleaning pipeline
    3. Generate processing statistics
    4. Save cleaned version with metadata
    
    Args:
        txt_path (str): Absolute path to the input text file (.txt)
        output_folder (str): Directory where cleaned text will be saved
        
    Returns:
        None: Function performs file I/O operations and logs progress to console
        
    Note:
        Output filename format: '{original_name}_cleaned.txt'
        Includes word count and character statistics in console output
    """
    # Step 1: Ensure output directory exists
    print(f"📁 Creating output directory: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 2: Read the text file
    print(f"📄 Processing text file: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except Exception as e:
        print(f"❌ Error reading {txt_path}: {e}")
        return
    
    # Step 3: Check if file has content
    if not raw_text.strip():
        print(f"⚠️  Warning: No content found in {txt_path}")
        return
    
    # Step 4: Clean the text for LLM processing
    print(f"  → Cleaning text for LLM processing...")
    cleaned_text = clean_text_for_llm(raw_text)
    
    # Step 5: Generate output filename
    txt_name = Path(txt_path).stem
    output_path = os.path.join(output_folder, f"{txt_name}_cleaned.txt")
    print(f"  → Saving to: {output_path}")
    
    # Step 6: Save cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Step 7: Display success message with statistics
    print(f"✅ Successfully saved cleaned text to: {output_path}")
    print(f"   📊 Word count: {len(cleaned_text.split())} words")
    print(f"   📏 Character count: {len(cleaned_text)} characters")

def process_pdf_to_txt(pdf_path, output_folder="output_data/cleaned_data"):
    """
    Complete PDF-to-text processing pipeline with intelligent extraction and cleaning.
    
    This is the main PDF processing function that orchestrates the entire extraction
    and cleaning workflow. It automatically selects the best extraction method based
    on content quality and applies comprehensive cleaning for LLM optimization.
    
    Processing pipeline:
    1. Attempt multi-method text extraction (PyMuPDF → PyPDF2 → OCR)
    2. Evaluate extraction quality and select best result
    3. Apply advanced text cleaning for LLM processing
    4. Generate comprehensive statistics (pages, words, characters)
    5. Save cleaned text with descriptive filename
    
    Args:
        pdf_path (str): Absolute path to the input PDF file
        output_folder (str): Directory where cleaned text will be saved
        
    Returns:
        None: Function performs file I/O operations and logs progress to console
              
    Output format:
        - Filename: '{pdf_name}_cleaned.txt'
        - Content: LLM-optimized text ready for chunking and analysis
        - Statistics: Page count, word count, character count logged to console
        
    Note:
        The function handles various PDF types automatically:
        - Standard PDFs with extractable text
        - Complex layouts requiring PyMuPDF
        - Scanned documents requiring OCR
        - Password-protected files (will fail gracefully)
    """
    # Step 1: Ensure output directory exists
    print(f"📁 Creating output directory: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 2: Extract text from PDF using multiple methods
    print(f"📄 Processing: {pdf_path}")
    print(f"  → Attempting text extraction...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Step 3: Check if extraction was successful
    if not raw_text.strip():
        print(f"⚠️  Warning: No text extracted from {pdf_path}")
        print(f"  → This might be a scanned PDF without OCR libraries")
        return
    
    # Step 4: Clean and process the extracted text for LLM
    print(f"  → Cleaning extracted text for LLM processing...")
    cleaned_text = clean_text_for_llm(raw_text)
    
    # Step 5: Generate output filename (PDF name + _cleaned.txt extension)
    pdf_name = Path(pdf_path).stem
    output_path = os.path.join(output_folder, f"{pdf_name}_cleaned.txt")
    print(f"  → Saving to: {output_path}")
    
    # Step 6: Save cleaned text (no metadata headers for LLM processing)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Step 7: Display success message with statistics
    print(f"✅ Successfully saved cleaned text to: {output_path}")
    print(f"   📊 Word count: {len(cleaned_text.split())} words")
    print(f"   📏 Character count: {len(cleaned_text)} characters")
    if '===' in cleaned_text:
        page_count = cleaned_text.count('=== Page')
        print(f"   📄 Pages processed: {page_count}")
    if '(OCR)' in cleaned_text:
        print(f"   🔍 OCR was used for scanned content")

def process_all_files_in_folders(pdf_folder="pdf_data", txt_folder="input_data", output_folder="output_data/cleaned_data"):
    """
    Batch processing pipeline for multiple PDF and text files across directories.
    
    This function provides a comprehensive batch processing solution for preparing
    large document collections for knowledge graph construction. It processes all
    supported file types from multiple input directories and generates a unified
    collection of cleaned, LLM-ready text files.
    
    Processing workflow:
    1. Scan specified directories for supported file types
    2. Process each PDF using intelligent extraction methods
    3. Process each text file using consistent cleaning pipeline
    4. Generate comprehensive processing statistics
    5. Provide detailed success/failure reporting
    
    Supported file types:
    - PDF files (.pdf): All types including scanned documents
    - Text files (.txt): Pre-existing text content
    
    Args:
        pdf_folder (str): Directory path containing PDF files to process
                         Default: 'pdf_data' (relative to current directory)
        txt_folder (str): Directory path containing text files to process
                         Default: 'input_data' (relative to current directory)
        output_folder (str): Directory where all cleaned files will be saved
                           Default: 'output_data/cleaned_data'
        
    Returns:
        tuple: (successful_count, failed_count) - Processing summary statistics
              
    Note:
        - Creates output directory automatically if it doesn't exist
        - Skips files that cause processing errors (logs errors to console)
        - Maintains original filename structure with '_cleaned' suffix
        - Provides real-time progress updates during batch processing
    """
    # Step 1: Ensure output directory exists
    print(f"📁 Creating output directory: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    total_files = 0
    successful_count = 0
    failed_count = 0
    
    # Step 2: Process PDF files from pdf_data folder
    print(f"\n🔍 Processing PDF files from: {pdf_folder}")
    if os.path.exists(pdf_folder):
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        if pdf_files:
            print(f"📋 Found {len(pdf_files)} PDF files:")
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"   {i}. {pdf_file}")
            
            for i, pdf_file in enumerate(pdf_files, 1):
                total_files += 1
                print(f"\n📄 Processing PDF {i}/{len(pdf_files)}: {pdf_file}")
                pdf_path = os.path.join(pdf_folder, pdf_file)
                
                try:
                    process_pdf_to_txt(pdf_path, output_folder)
                    successful_count += 1
                    print(f"✅ PDF {i} completed successfully")
                except Exception as e:
                    print(f"❌ Error processing {pdf_file}: {e}")
                    failed_count += 1
        else:
            print(f"📭 No PDF files found in '{pdf_folder}'")
    else:
        print(f"⚠️  PDF folder '{pdf_folder}' does not exist")
    
    # Step 3: Process text files from input_data folder
    print(f"\n🔍 Processing text files from: {txt_folder}")
    if os.path.exists(txt_folder):
        txt_files = [f for f in os.listdir(txt_folder) if f.lower().endswith('.txt')]
        if txt_files:
            print(f"📋 Found {len(txt_files)} text files:")
            for i, txt_file in enumerate(txt_files, 1):
                print(f"   {i}. {txt_file}")
            
            for i, txt_file in enumerate(txt_files, 1):
                total_files += 1
                print(f"\n📄 Processing text file {i}/{len(txt_files)}: {txt_file}")
                txt_path = os.path.join(txt_folder, txt_file)
                
                try:
                    process_text_file(txt_path, output_folder)
                    successful_count += 1
                    print(f"✅ Text file {i} completed successfully")
                except Exception as e:
                    print(f"❌ Error processing {txt_file}: {e}")
                    failed_count += 1
        else:
            print(f"📭 No text files found in '{txt_folder}'")
    else:
        print(f"⚠️  Text folder '{txt_folder}' does not exist")
    
    # Step 4: Display final summary
    print(f"\n📊 Processing complete!")
    print(f"   📁 Total files processed: {total_files}")
    print(f"   ✅ Successfully processed: {successful_count} files")
    if failed_count > 0:
        print(f"   ❌ Failed to process: {failed_count} files")
    print(f"   💾 Cleaned output saved to: {output_folder}")
    
    return successful_count, failed_count

def process_all_pdfs_in_folder(pdf_folder="pdf_data", output_folder="output_data/cleaned_data"):
    """
    Legacy function - kept for backward compatibility.
    Use process_all_files_in_folders for enhanced functionality.
    """
    return process_all_files_in_folders(pdf_folder=pdf_folder, output_folder=output_folder)

if __name__ == "__main__":
    # Process all files from both pdf_data and input_data folders
    print("Enhanced PDF & Text Processor for LLM Pipeline")
    print("=" * 50)
    print("Processing files from 'pdf_data' and 'input_data' folders...")
    print("Cleaned output will be saved to 'output_data/cleaned_data' folder\n")
    
    # Process all files in both folders
    successful, failed = process_all_files_in_folders()
    
    print("\n✅ Processing complete!")
    print(f"📁 Check the 'output_data/cleaned_data' folder for cleaned text files.")
    print(f"🚀 Files are now optimized for LLM chunking and entity extraction!")
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed to process. Check the logs above for details.")