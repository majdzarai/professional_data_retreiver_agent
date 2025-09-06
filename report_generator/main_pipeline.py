#!/usr/bin/env python3
"""
Main Pipeline Script
Runs the planner and researcher agents sequentially.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_planner(topic, output_dir):
    """
    Run the planner agent.
    
    Args:
        topic (str): The topic for the report
        output_dir (str): Output directory for planner results
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Planner Agent...")
    
    try:
        # Run the planner
        cmd = [sys.executable, "main_planner.py", "--structure", "report_structure.json", "--topic", topic]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("Planner Agent completed successfully")
        logger.info(f"Planner output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Planner Agent failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running Planner Agent: {e}")
        return False

def run_researcher(planner_output_dir, output_dir):
    """
    Run the researcher agent.
    
    Args:
        planner_output_dir (str): Directory containing planner results
        output_dir (str): Output directory for researcher results
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Researcher Agent...")
    
    try:
        # Run the researcher
        cmd = [sys.executable, "main_researcher.py", "--planner-output", planner_output_dir, "--output-dir", output_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("Researcher Agent completed successfully")
        logger.info(f"Researcher output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Researcher Agent failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running Researcher Agent: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete report generation pipeline")
    parser.add_argument("--topic", required=True, help="Topic for the report")
    parser.add_argument("--output-dir", default="output", help="Base output directory (default: output)")
    
    args = parser.parse_args()
    
    # Setup output directories
    base_output_dir = Path(args.output_dir)
    planner_output_dir = base_output_dir / "planner"
    researcher_output_dir = base_output_dir / "researcher"
    
    # Create output directories
    planner_output_dir.mkdir(parents=True, exist_ok=True)
    researcher_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting pipeline for topic: {args.topic}")
    logger.info(f"Output directory: {base_output_dir.absolute()}")
    
    # Step 1: Run Planner
    if not run_planner(args.topic, str(planner_output_dir)):
        logger.error("Pipeline failed at Planner stage")
        sys.exit(1)
    
    # Step 2: Run Researcher
    if not run_researcher(str(planner_output_dir), str(researcher_output_dir)):
        logger.error("Pipeline failed at Researcher stage")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {base_output_dir.absolute()}")
    logger.info(f"- Planner output: {planner_output_dir.absolute()}")
    logger.info(f"- Researcher output: {researcher_output_dir.absolute()}")

if __name__ == "__main__":
    main()