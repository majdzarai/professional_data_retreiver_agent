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

def run_planner(topic, output_dir, structure_file=None):
    """
    Run the planner agent.
    
    Args:
        topic (str): The report topic
        output_dir (str): Output directory for planner results
        structure_file (str, optional): Path to structure file
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Planner Agent...")
    
    try:
        # Build command
        cmd = [sys.executable, "main_planner.py", "--topic", topic, "--output-dir", output_dir]
        if structure_file:
            cmd.extend(["--structure", structure_file])
        
        # Run the planner
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

def run_section_report_agent(researcher_output_dir: str, section_prompts_dir: str) -> bool:
    """
    Run the Section Report Agent to generate section prompts.
    
    Args:
        researcher_output_dir: Directory containing researcher output
        section_prompts_dir: Directory to save section prompts
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            sys.executable, "main_section_report.py",
            "--researcher-output", researcher_output_dir,
            "--output-dir", section_prompts_dir
        ]
        
        logger.info(f"Running Section Report Agent: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("Section Report Agent completed successfully")
            return True
        else:
            logger.error(f"Section Report Agent failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running Section Report Agent: {str(e)}")
        return False

def run_writer_agent(section_prompts_dir: str, writer_output_dir: str) -> bool:
    """
    Run the Writer Agent to generate actual content from section prompts.
    
    Args:
        section_prompts_dir: Directory containing section prompts
        writer_output_dir: Directory to save generated content
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            sys.executable, "main_writer.py",
            "--prompts-dir", section_prompts_dir,
            "--output-dir", writer_output_dir
        ]
        
        logger.info(f"Running Writer Agent: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("Writer Agent completed successfully")
            return True
        else:
            logger.error(f"Writer Agent failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running Writer Agent: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete report generation pipeline")
    parser.add_argument("--topic", required=True, help="Topic for the report")
    parser.add_argument("--structure", help="Path to report structure JSON file")
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
    if not run_planner(args.topic, str(planner_output_dir), args.structure):
        logger.error("Pipeline failed at Planner stage")
        sys.exit(1)
    
    # Step 2: Run Researcher
    if not run_researcher(str(planner_output_dir), str(researcher_output_dir)):
        logger.error("Pipeline failed at Researcher stage")
        sys.exit(1)
    
    # Step 3: Run Section Report Agent
    section_prompts_dir = base_output_dir / "section_prompts"
    section_prompts_dir.mkdir(parents=True, exist_ok=True)
    # The researcher creates a nested 'researcher' directory, so we need to point to the correct path
    actual_researcher_output = researcher_output_dir / "researcher"
    if not run_section_report_agent(str(actual_researcher_output), str(section_prompts_dir)):
        logger.error("Pipeline failed at Section Report stage")
        sys.exit(1)
    
    # Step 4: Run Writer Agent
    writer_output_dir = base_output_dir / "writer"
    writer_output_dir.mkdir(parents=True, exist_ok=True)
    if not run_writer_agent(str(section_prompts_dir), str(writer_output_dir)):
        logger.error("Pipeline failed at Writer stage")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {base_output_dir.absolute()}")
    logger.info(f"- Planner output: {planner_output_dir.absolute()}")
    logger.info(f"- Researcher output: {researcher_output_dir.absolute()}")
    logger.info(f"- Section prompts: {section_prompts_dir.absolute()}")
    logger.info(f"- Writer output: {writer_output_dir.absolute()}")

if __name__ == "__main__":
    main()