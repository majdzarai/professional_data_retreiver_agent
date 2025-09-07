#!/usr/bin/env python3
"""
Main Writer Script

This script runs the Writer Agent to generate actual content from section prompts.
It processes structured prompts created by the Section Report Agent and generates
well-written report sections organized by chapters.

Usage:
    python main_writer.py --prompts-dir <prompts_directory> --output-dir <output_directory>

Example:
    python main_writer.py --prompts-dir output/section_prompts --output-dir output/writer
"""

import argparse
import logging
import sys
from pathlib import Path
from writer_agent import WriterAgent

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate report content from section prompts using Writer Agent'
    )
    
    parser.add_argument(
        '--prompts-dir',
        type=str,
        required=True,
        help='Directory containing section prompts (output from Section Report Agent)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/writer',
        help='Output directory for generated content (default: output/writer)'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the Writer Agent."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate input directory
        prompts_dir = Path(args.prompts_dir)
        if not prompts_dir.exists():
            logger.error(f"Prompts directory not found: {prompts_dir}")
            sys.exit(1)
        
        # Check for prompts summary file
        summary_file = prompts_dir / "prompts_summary.json"
        if not summary_file.exists():
            logger.error(f"Prompts summary file not found: {summary_file}")
            logger.error("Make sure to run the Section Report Agent first to generate prompts.")
            sys.exit(1)
        
        logger.info(f"Processing section prompts from: {prompts_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Initialize Writer Agent
        writer = WriterAgent(output_dir=args.output_dir)
        
        # Process section prompts
        success = writer.process_section_prompts(str(prompts_dir))
        
        if success:
            logger.info("Content generation completed successfully!")
            logger.info(f"Total chapters generated: {len(writer.chapters)}")
            logger.info(f"Total sections generated: {len(writer.generated_sections)}")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info(f"Sections directory: {writer.sections_dir}")
            logger.info(f"Chapters directory: {writer.chapters_dir}")
            
            # Display summary
            print("\n" + "="*60)
            print("CONTENT GENERATION COMPLETED")
            print("="*60)
            
            total_words = sum(section['word_count'] for section in writer.generated_sections)
            print(f"Total chapters: {len(writer.chapters)}")
            print(f"Total sections: {len(writer.generated_sections)}")
            print(f"Total word count: {total_words:,} words")
            print(f"Output directory: {args.output_dir}")
            
            print("\nGenerated chapters:")
            for chapter_title, sections in writer.chapters.items():
                chapter_words = sum(s['word_count'] for s in sections)
                print(f"  - {chapter_title}")
                print(f"    Sections: {len(sections)}")
                print(f"    Word count: {chapter_words:,} words")
                for section in sections:
                    print(f"      • {section['section_title']} ({section['word_count']} words)")
            
            print("\nNext steps:")
            print("1. Review the generated content in the output directory")
            print("2. Check individual sections in the 'sections' subdirectory")
            print("3. Review complete chapters in the 'chapters' subdirectory")
            print("4. Use the markdown files for easy reading and editing")
            
        else:
            logger.error("Content generation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Content generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()