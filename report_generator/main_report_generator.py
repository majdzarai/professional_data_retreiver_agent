#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Report Generator

This script orchestrates the multi-agent report generation process:
1. Takes a report structure as input
2. Uses the Planner Agent to generate questions and RAG queries
3. Uses the Researcher Agent to run queries against the RAG system
4. Uses the Writer Agent to generate answers based on retrieved evidence
5. Uses the Editor Agent to compile the final report
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import from rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agents
from report_generator.agents.planner_agent import PlannerAgent
from report_generator.agents.researcher_agent import ResearcherAgent
from report_generator.agents.writer_agent import WriterAgent
from report_generator.agents.editor_agent import EditorAgent

# Import utilities
from report_generator.utils.config import ReportGeneratorConfig
from report_generator.utils.logger import setup_logger


class ReportGenerator:
    """Main class for orchestrating the report generation process."""

    def __init__(self, config: ReportGeneratorConfig):
        """Initialize the report generator with configuration.

        Args:
            config: Configuration object for the report generator
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create output directories if they don't exist
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize agents
        self.planner = PlannerAgent(config)
        self.researcher = ResearcherAgent(config)
        self.writer = WriterAgent(config)
        self.editor = EditorAgent(config)
        
        # Initialize storage for intermediate results
        self.report_structure = None
        self.section_plans = {}
        self.retrieved_evidence = {}
        self.section_drafts = {}
        self.final_report = None

    def load_report_structure(self, structure_file: str) -> Dict:
        """Load the report structure from a JSON file.

        Args:
            structure_file: Path to the JSON file containing the report structure

        Returns:
            The loaded report structure as a dictionary
        """
        self.logger.info(f"Loading report structure from {structure_file}")
        with open(structure_file, 'r') as f:
            self.report_structure = json.load(f)
        return self.report_structure

    def generate_section_plans(self) -> Dict:
        """
Generate plans for each section using the Planner Agent.

        Returns:
            Dictionary mapping section IDs to their plans
        """
        self.logger.info("Generating section plans")
        for chapter in self.report_structure['chapters']:
            chapter_id = chapter['id']
            chapter_title = chapter['title']
            
            # Process the chapter itself as a section
            self.logger.info(f"Planning section: {chapter_title} ({chapter_id})")
            section_plan = self.planner.generate_plan(chapter)
            self.section_plans[chapter_id] = section_plan
            
            # Save the plan to disk
            plan_file = self.output_dir / f"plan_{chapter_id}.json"
            with open(plan_file, 'w') as f:
                json.dump(section_plan, f, indent=2)
            
            # Process each section within the chapter if any
            if 'sections' in chapter and chapter['sections']:
                for section in chapter['sections']:
                    section_id = section['id']
                    section_title = section['title']
                    self.logger.info(f"Planning subsection: {section_title} ({section_id})")
                    
                    # Add chapter information to the section
                    section['chapter_id'] = chapter_id
                    section['chapter_title'] = chapter_title
                    
                    # Generate plan for this section
                    section_plan = self.planner.generate_plan(section)
                    self.section_plans[section_id] = section_plan
                    
                    # Save the plan to disk
                    plan_file = self.output_dir / f"plan_{section_id}.json"
                    with open(plan_file, 'w') as f:
                        json.dump(section_plan, f, indent=2)
        
        return self.section_plans

    def retrieve_evidence(self) -> Dict:
        """Retrieve evidence for each question using the Researcher Agent.

        Returns:
            Dictionary mapping question IDs to their evidence
        """
        self.logger.info("Retrieving evidence for questions")
        
        # Check if we have a plan with 'sections' key (from planner.generate_plan)
        if 'sections' in self.section_plans:
            # Process each section in the plan
            for section in self.section_plans['sections']:
                if 'questions' in section:
                    for question in section['questions']:
                        question_id = question['qid']
                        self.logger.info(f"Researching question: {question_id}")
                        
                        # Retrieve evidence for this question
                        evidence = self.researcher.retrieve_evidence(question)
                        self.retrieved_evidence[question_id] = evidence
                        
                        # Save the evidence to disk
                        evidence_file = self.output_dir / f"evidence_{question_id}.json"
                        with open(evidence_file, 'w') as f:
                            json.dump(evidence, f, indent=2)
        else:
            # Process each section plan individually
            for section_id, section_plan in self.section_plans.items():
                if 'questions' in section_plan:
                    for question in section_plan['questions']:
                        question_id = question['qid']
                        self.logger.info(f"Researching question: {question_id}")
                        
                        # Retrieve evidence for this question
                        evidence = self.researcher.retrieve_evidence(question)
                        self.retrieved_evidence[question_id] = evidence
                        
                        # Save the evidence to disk
                        evidence_file = self.output_dir / f"evidence_{question_id}.json"
                        with open(evidence_file, 'w') as f:
                            json.dump(evidence, f, indent=2)
        
        return self.retrieved_evidence

    def generate_section_drafts(self) -> Dict:
        """Generate drafts for each section using the Writer Agent.

        Returns:
            Dictionary mapping section IDs to their drafts
        """
        self.logger.info("Generating section drafts")
        
        # Prepare research results in the format expected by the Writer Agent
        research_results = {"sections": []}
        
        # Check if we have a plan with 'sections' key (from planner.generate_plan)
        if 'sections' in self.section_plans:
            # Process each section in the plan
            for section in self.section_plans['sections']:
                section_id = section.get('section_id', '')
                if not section_id:
                    continue
                    
                self.logger.info(f"Writing draft for section: {section_id}")
                
                # Create a section entry for the research results
                section_entry = {
                    "section_id": section_id,
                    "section_title": section.get('section_title', ''),
                    "questions": []
                }
                
                # Add questions and evidence
                if 'questions' in section:
                    for question in section['questions']:
                        question_id = question['qid']
                        question_entry = {
                            "qid": question_id,
                            "text": question.get('text', ''),
                            "evidence": self.retrieved_evidence.get(question_id, [])
                        }
                        section_entry["questions"].append(question_entry)
                
                research_results["sections"].append(section_entry)
        else:
            # Process each section plan individually
            for section_id, section_plan in self.section_plans.items():
                self.logger.info(f"Writing draft for section: {section_id}")
                
                # Create a section entry for the research results
                section_entry = {
                    "section_id": section_id,
                    "section_title": section_plan.get('section_title', ''),
                    "questions": []
                }
                
                # Add questions and evidence
                if 'questions' in section_plan:
                    for question in section_plan['questions']:
                        question_id = question['qid']
                        question_entry = {
                            "qid": question_id,
                            "text": question.get('text', ''),
                            "evidence": self.retrieved_evidence.get(question_id, [])
                        }
                        section_entry["questions"].append(question_entry)
                
                research_results["sections"].append(section_entry)
        
        # Generate drafts for all sections
        section_drafts = self.writer.write_sections(research_results)
        
        # Save drafts to disk
        for section in section_drafts.get('sections', []):
            section_id = section.get('section_id')
            content = section.get('content')
            
            if section_id and content:
                self.section_drafts[section_id] = content
                
                # Save the draft to disk
                draft_file = self.output_dir / f"draft_{section_id}.md"
                with open(draft_file, 'w') as f:
                    f.write(content)
        
        return self.section_drafts

    def compile_final_report(self) -> str:
        """Compile the final report using the Editor Agent.

        Returns:
            The final report as a string
        """
        self.logger.info("Compiling final report")
        
        # Prepare section drafts in the format expected by the Editor Agent
        formatted_section_drafts = {
            "report_title": self.report_structure.get("title", "Untitled Report"),
            "sections": []
        }
        
        # Convert self.section_drafts (dict of section_id -> content) to the expected format
        for section_id, content in self.section_drafts.items():
            # Find the section in the report structure to get the title
            section_title = ""
            for chapter in self.report_structure.get("chapters", []):
                for section in chapter.get("sections", []):
                    if section.get("id") == section_id:
                        section_title = section.get("title", "")
                        break
            
            formatted_section_drafts["sections"].append({
                "section_id": section_id,
                "section_title": section_title,
                "content": content
            })
        
        self.final_report = self.editor.compile_report(formatted_section_drafts)
        
        # Save the final report to disk
        report_file = self.output_dir / "final_report.md"
        with open(report_file, 'w') as f:
            f.write(self.final_report)
        
        return self.final_report

    def run(self, structure_file: str) -> str:
        """Run the complete report generation process.

        Args:
            structure_file: Path to the JSON file containing the report structure

        Returns:
            Path to the generated report
        """
        start_time = time.time()
        self.logger.info("Starting report generation process")
        
        # Step 1: Load report structure
        self.load_report_structure(structure_file)
        
        # Step 2: Generate section plans
        self.generate_section_plans()
        
        # Step 3: Retrieve evidence
        self.retrieve_evidence()
        
        # Step 4: Generate section drafts
        self.generate_section_drafts()
        
        # Step 5: Compile final report
        self.compile_final_report()
        
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"Report generation completed in {duration:.2f} seconds")
        
        report_file = self.output_dir / "final_report.md"
        self.logger.info(f"Final report saved to: {report_file}")
        
        return str(report_file)


def main():
    """Main entry point for the report generator."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a report using the multi-agent framework")
    parser.add_argument('--structure', '-s', required=True,
                       help='Path to the JSON file containing the report structure')
    parser.add_argument('--output-dir', '-o', default='output/reports',
                       help='Directory to save the generated report and intermediate files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--model', default='gpt-4',
                       help='LLM model to use for the agents')
    parser.add_argument('--rag-path', default='../rag',
                       help='Path to the RAG pipeline directory')
    parser.add_argument('--llama-model-path',
                       help='Path to the local Llama model file')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level)
    
    # Handle Llama model configuration
    if 'llama' in args.model.lower():
        # If using llama-3.1 format, convert to llama3.1 for Ollama
        if args.model.lower() in ['llama-3.1', 'llama-3.1-8b']:
            args.model = 'llama3.1'
            logging.info(f"Using Ollama model: {args.model}")
        
        # Set model path if provided (for llama-cpp-python)
        if args.llama_model_path:
            os.environ["LLAMA_MODEL_PATH"] = args.llama_model_path
            logging.info(f"Using Llama model at: {args.llama_model_path}")
    
    # Create configuration
    config = ReportGeneratorConfig(
        output_dir=args.output_dir,
        model=args.model,
        rag_path=args.rag_path,
        verbose=args.verbose
    )
    
    # Create and run the report generator
    generator = ReportGenerator(config)
    report_file = generator.run(args.structure)
    
    print(f"\n🎉 Report generation completed!")
    print(f"📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()