#!/usr/bin/env python
import argparse
import logging
import sys
from pathlib import Path
from planner_agent import PlannerAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Generate a plan for a report based on a topic.')
    parser.add_argument('--structure', required=True, help='Path to the report structure JSON file')
    parser.add_argument('--topic', required=True, help='Topic of the report')
    parser.add_argument('--output-dir', default='output/planner', help='Output directory for planner results')
    
    return parser.parse_args()

def main():
    """Main entry point for the planner CLI."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate the structure file path
    structure_path = Path(args.structure)
    if not structure_path.exists():
        logger.error(f"Structure file not found: {structure_path}")
        sys.exit(1)
    
    try:
        # Initialize the planner agent
        logger.info(f"Initializing planner agent with structure: {structure_path}")
        logger.info(f"Output directory: {args.output_dir}")
        planner = PlannerAgent(structure_path, args.output_dir)
        
        # Generate the plan
        logger.info(f"Generating plan for topic: {args.topic}")
        plan = planner.generate_plan(args.topic)
        
        # Log the completion
        logger.info(f"Plan generation completed. Output saved to {planner.output_dir}")
        logger.info(f"Global plan: {planner.output_dir / 'plan.json'}")
        logger.info(f"Section plans: {planner.sections_dir}/*.json")
        
        # Print a summary
        print(f"\nPlan generation completed successfully!")
        print(f"Generated plans for {len(plan['sections'])} sections.")
        print(f"\nOutput files:")
        print(f"  - Global plan: {planner.output_dir / 'plan.json'}")
        print(f"  - Section plans: {planner.sections_dir}/*.json")
        
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()