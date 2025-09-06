#!/usr/bin/env python3
"""
Main script for running the Researcher Agent.

This script processes planner output and enriches it with research data
from the RAG pipeline, generating structured JSON outputs.

Usage:
    python main_researcher.py --planner-output output/planner --output-dir output --top-k 5
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from researcher_agent import ResearcherAgent

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the Researcher Agent to enrich planner output with research data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_researcher.py --planner-output output/planner --output-dir output
  python main_researcher.py --planner-output output/planner --output-dir output --top-k 10

Output:
  - Global research: {output-dir}/researcher/research.json
  - Section research: {output-dir}/researcher/sections/*.json
        """
    )
    
    parser.add_argument(
        "--planner-output",
        type=str,
        required=True,
        help="Directory containing planner output files (should contain plan.json and sections/ folder)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for researcher results (will create researcher/ subfolder)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve for each query (default: 5)"
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
        args: Parsed arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Check if planner output directory exists
    planner_dir = Path(args.planner_output)
    if not planner_dir.exists():
        print(f"Error: Planner output directory does not exist: {planner_dir}")
        sys.exit(1)
    
    # Check if planner output has required structure
    plan_file = planner_dir / "plan.json"
    sections_dir = planner_dir / "sections"
    
    if not plan_file.exists():
        print(f"Error: Global plan file not found: {plan_file}")
        sys.exit(1)
    
    if not sections_dir.exists():
        print(f"Error: Sections directory not found: {sections_dir}")
        sys.exit(1)
    
    # Check if sections directory has JSON files
    json_files = list(sections_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in sections directory: {sections_dir}")
        sys.exit(1)
    
    # Validate top-k parameter
    if args.top_k <= 0:
        print(f"Error: top-k must be a positive integer, got: {args.top_k}")
        sys.exit(1)
    
    print(f"Validation passed:")
    print(f"  - Planner output: {planner_dir}")
    print(f"  - Found {len(json_files)} section files")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Top-k results: {args.top_k}")

def setup_logging(log_level):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level string
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """
    Main function to run the researcher agent.
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Researcher Agent...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Validate arguments
        validate_arguments(args)
        
        # Initialize and run researcher agent
        researcher = ResearcherAgent(
            planner_output_dir=args.planner_output,
            output_dir=args.output_dir,
            top_k=args.top_k
        )
        
        # Run the research process
        researcher.run_research()
        
        logger.info("Researcher Agent completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Researcher Agent failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()