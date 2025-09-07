#!/usr/bin/env python3
"""
Main script for running the PDF Generator.

This script processes writer output and generates a comprehensive PDF report
with all chapters in the correct order according to the report structure.

Usage:
    python main_pdf_generator.py --writer-output output/writer --structure report_structure.json --output-dir output/pdf
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_generator import PDFGenerator

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate PDF report from writer output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate PDF from writer output
    python main_pdf_generator.py --writer-output output/writer --structure report_structure.json
    
    # Specify custom output directory
    python main_pdf_generator.py --writer-output output/writer --structure report_structure.json --output-dir output/pdf
        """
    )
    
    parser.add_argument(
        "--writer-output",
        required=True,
        help="Directory containing writer output (chapters and report summary)"
    )
    
    parser.add_argument(
        "--structure",
        required=True,
        help="Path to report structure JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output/pdf",
        help="Output directory for PDF file (default: output/pdf)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    return parser.parse_args()

def setup_logging(log_level: str):
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Check writer output directory
    writer_path = Path(args.writer_output)
    if not writer_path.exists():
        print(f"Error: Writer output directory does not exist: {writer_path}")
        sys.exit(1)
    
    # Check for required directories
    chapters_dir = writer_path / "chapters"
    if not chapters_dir.exists():
        print(f"Error: Chapters directory not found: {chapters_dir}")
        sys.exit(1)
    
    # Check if chapters directory has any markdown files
    chapter_files = list(chapters_dir.glob("*.md"))
    if not chapter_files:
        print(f"Error: No chapter markdown files found in: {chapters_dir}")
        sys.exit(1)
    
    print(f"Found {len(chapter_files)} chapter files to process")
    
    # Check report structure file
    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Report structure file does not exist: {structure_path}")
        sys.exit(1)
    
    # Check if structure file is valid JSON
    try:
        import json
        with open(structure_path, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        
        if "chapters" not in structure:
            print(f"Error: Invalid report structure - missing 'chapters' key")
            sys.exit(1)
            
        print(f"Report structure contains {len(structure['chapters'])} chapters")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in structure file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading structure file: {e}")
        sys.exit(1)

def main():
    """
    Main entry point for the PDF generator CLI.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    validate_arguments(args)
    
    try:
        # Initialize the PDF generator
        logger.info(f"Initializing PDF Generator with output directory: {args.output_dir}")
        generator = PDFGenerator(output_dir=args.output_dir)
        
        # Generate PDF report
        logger.info(f"Generating PDF from writer output: {args.writer_output}")
        pdf_path = generator.generate_pdf_report(args.writer_output, args.structure)
        
        # Log the completion
        logger.info("PDF generation completed successfully!")
        logger.info(f"PDF file saved to: {pdf_path}")
        
        # Print a summary
        print("\n" + "="*60)
        print("PDF GENERATION COMPLETED")
        print("="*60)
        print(f"PDF file: {pdf_path}")
        print(f"Output directory: {args.output_dir}")
        print("\nNext steps:")
        print(f"1. Open the PDF file: {pdf_path}")
        print("2. Review the generated report")
        print("3. Share with stakeholders as needed")
        
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()