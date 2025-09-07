#!/usr/bin/env python3
"""
Main script for running the Section Report Agent.

This script processes researcher output and generates structured writing prompts
for each section that can be consumed by a writer agent.

Usage:
    python main_section_report.py --researcher-output output/researcher --output-dir output/section_prompts
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from section_report_agent import SectionReportAgent

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the Section Report Agent to generate writing prompts from researcher output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_section_report.py --researcher-output output/researcher
  python main_section_report.py --researcher-output output/researcher --output-dir output/section_prompts

Output:
  - Prompts summary: {output-dir}/prompts_summary.json
  - Individual prompts: {output-dir}/prompts/*.json
        """
    )
    
    parser.add_argument(
        "--researcher-output",
        type=str,
        required=True,
        help="Directory containing researcher output files (should contain research.json and sections/ folder)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/section_prompts",
        help="Directory to save generated prompts (default: output/section_prompts)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Check researcher output directory
    researcher_path = Path(args.researcher_output)
    if not researcher_path.exists():
        print(f"Error: Researcher output directory does not exist: {researcher_path}")
        sys.exit(1)
    
    # Check for required files
    research_file = researcher_path / "research.json"
    if not research_file.exists():
        print(f"Error: Research file not found: {research_file}")
        sys.exit(1)
    
    sections_dir = researcher_path / "sections"
    if not sections_dir.exists():
        print(f"Error: Sections directory not found: {sections_dir}")
        sys.exit(1)
    
    # Check if sections directory has any JSON files
    section_files = list(sections_dir.glob("*.json"))
    if not section_files:
        print(f"Error: No section files found in: {sections_dir}")
        sys.exit(1)
    
    print(f"Found {len(section_files)} section files to process")

def setup_logging(log_level):
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """
    Main entry point for the section report agent CLI.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    validate_arguments(args)
    
    try:
        # Initialize the section report agent
        logger.info(f"Initializing Section Report Agent with output directory: {args.output_dir}")
        agent = SectionReportAgent(output_dir=args.output_dir)
        
        # Process researcher output
        logger.info(f"Processing researcher output from: {args.researcher_output}")
        summary = agent.process_researcher_output(args.researcher_output)
        
        # Log the completion
        logger.info("Section report generation completed successfully!")
        logger.info(f"Total sections processed: {summary['total_sections']}")
        logger.info(f"Output directory: {summary['output_directory']}")
        logger.info(f"Prompts summary: {Path(summary['output_directory']) / 'prompts_summary.json'}")
        
        # Print a summary
        print("\n" + "="*60)
        print("SECTION REPORT GENERATION COMPLETED")
        print("="*60)
        print(f"Topic: {summary['topic']}")
        print(f"Sections processed: {summary['total_sections']}")
        print(f"Output directory: {summary['output_directory']}")
        print("\nGenerated prompts:")
        
        for section_name, details in summary['generated_prompts'].items():
            print(f"  - {details['chapter_title']} > {details['section_title']}")
            print(f"    Questions: {details['questions_count']}")
            print(f"    Prompt file: {details['prompt_file']}")
            print()
        
        print("Next steps:")
        print("1. Review the generated prompts in the output directory")
        print("2. Use these prompts with a writer agent to generate the actual report sections")
        print("3. Combine all sections into a final report")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()