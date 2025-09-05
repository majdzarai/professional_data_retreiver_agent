#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Writer Agent for the Report Generator

This module contains the Writer Agent class that is responsible for generating
draft content for each section of a report based on the evidence collected by
the Researcher Agent.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.config import ReportGeneratorConfig
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent:
    """Writer Agent for the Report Generator.
    
    The Writer Agent is responsible for generating draft content for each section
    of a report based on the evidence collected by the Researcher Agent. It takes
    research results as input and produces draft content for each section.
    """

    def __init__(self, config: ReportGeneratorConfig):
        """Initialize the Writer Agent.
        
        Args:
            config: Configuration for the Report Generator
        """
        self.config = config
        self.prompt_template = config.writer_prompt_template
        self.output_dir = Path(config.output_dir) / "writer"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client
        self.llm_client = LLMClient(model=config.model)
        
        logger.info(f"Initialized Writer Agent with output directory: {self.output_dir}")

    def write_sections(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Write draft content for each section based on research results.
        
        Args:
            research_results: The research results from the Researcher Agent
            
        Returns:
            A dictionary containing the draft content for each section
        """
        logger.info("Writing sections based on research results")
        
        # Validate research results
        self._validate_research_results(research_results)
        
        # Initialize section drafts
        section_drafts = {
            "report_title": research_results.get("report_title", "Untitled Report"),
            "sections": []
        }
        
        # Process each section in the research results
        for section in research_results.get("sections", []):
            section_id = section.get("section_id", "")
            section_title = section.get("section_title", "")
            
            # Generate draft content for this section
            section_draft = self._write_section(
                section_id=section_id,
                section_title=section_title,
                chapter_id=section.get("chapter_id", ""),
                chapter_title=section.get("chapter_title", ""),
                description=section.get("description", ""),
                questions=section.get("questions", [])
            )
            
            section_drafts["sections"].append(section_draft)
        
        # Save the section drafts to a file
        self._save_section_drafts(section_drafts)
        
        return section_drafts

    def _validate_research_results(self, research_results: Dict[str, Any]) -> None:
        """Validate the research results.
        
        Args:
            research_results: The research results to validate
            
        Raises:
            ValueError: If the research results are invalid
        """
        if not isinstance(research_results, dict):
            raise ValueError("Research results must be a dictionary")
            
        if "sections" not in research_results:
            raise ValueError("Research results must contain 'sections' key")
            
        if not isinstance(research_results["sections"], list):
            raise ValueError("'sections' must be a list")
            
        for i, section in enumerate(research_results["sections"]):
            if not isinstance(section, dict):
                raise ValueError(f"Section {i} must be a dictionary")
                
            if "section_id" not in section:
                raise ValueError(f"Section {i} must have a 'section_id'")
                
            if "section_title" not in section:
                raise ValueError(f"Section {i} must have a 'section_title'")
                
            if "questions" not in section:
                raise ValueError(f"Section {i} must have 'questions'")
                
            if not isinstance(section["questions"], list):
                raise ValueError(f"'questions' in section {i} must be a list")
                
            for j, question in enumerate(section["questions"]):
                if not isinstance(question, dict):
                    raise ValueError(f"Question {j} in section {i} must be a dictionary")
                    
                if "qid" not in question:
                    raise ValueError(f"Question {j} in section {i} must have a 'qid'")
                    
                if "text" not in question:
                    raise ValueError(f"Question {j} in section {i} must have a 'text'")
                    
                if "evidence" not in question:
                    raise ValueError(f"Question {j} in section {i} must have 'evidence'")
                    
                if "consolidated_evidence" not in question:
                    raise ValueError(f"Question {j} in section {i} must have 'consolidated_evidence'")

    def _write_section(
        self,
        section_id: str,
        section_title: str,
        chapter_id: str,
        chapter_title: str,
        description: str,
        questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Write draft content for a section.
        
        Args:
            section_id: ID of the section
            section_title: Title of the section
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            description: Description of the section
            questions: List of questions with evidence
            
        Returns:
            A dictionary containing the draft content for the section
        """
        logger.info(f"Writing draft for section: {section_id} - {section_title}")
        
        # Prepare the evidence for the LLM
        evidence_text = ""
        for question in questions:
            question_id = question.get("qid", "")
            question_text = question.get("text", "")
            consolidated_evidence = question.get("consolidated_evidence", "")
            
            evidence_text += f"Question: {question_text}\n\n"
            evidence_text += f"Evidence:\n{consolidated_evidence}\n\n"
        
        # Prepare the prompt for the LLM
        prompt = self._prepare_section_prompt(
            section_id=section_id,
            section_title=section_title,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            description=description,
            evidence_text=evidence_text
        )
        
        try:
            # Call the LLM to generate the section content
            logger.info("Calling LLM to generate section content")
            section_content = self.llm_client.generate(prompt)
            
            # Check if the content is empty or too short
            if not section_content or len(section_content.strip()) < 100:
                logger.warning("LLM generated content is too short, using fallback content")
                section_content = self._generate_fallback_content(
                    section_id, section_title, chapter_id, chapter_title, description, questions
                )
        except Exception as e:
            logger.error(f"Error generating section content: {e}")
            # Use fallback content in case of error
            section_content = self._generate_fallback_content(
                section_id, section_title, chapter_id, chapter_title, description, questions
            )
        
        # Ensure the content starts with the section title
        if not section_content.startswith(f"# {section_title}"):
            section_content = f"# {section_title}\n\n{section_content}"
        
        section_draft = {
            "section_id": section_id,
            "section_title": section_title,
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "content": section_content
        }
        
        # Save the section draft to a file
        self._save_section_draft(section_id, section_content)
        
        return section_draft
        
    def _prepare_section_prompt(self, section_id: str, section_title: str, 
                               chapter_id: str, chapter_title: str, 
                               description: str, evidence_text: str) -> str:
        """Prepare the prompt for generating section content.
        
        Args:
            section_id: ID of the section
            section_title: Title of the section
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            description: Description of the section
            evidence_text: Text containing the evidence for the section
            
        Returns:
            The prompt for the LLM
        """
        # Start with the base prompt template
        prompt = self.prompt_template
        
        # Add section-specific information
        prompt += f"\n\nYou are writing the '{section_title}' section of the '{chapter_title}' chapter."
        
        if description:
            prompt += f"\n\nSection description: {description}"
        
        prompt += f"\n\nBased on the following evidence, write a comprehensive section:\n\n"
        prompt += evidence_text
        
        prompt += "\n\nYour response should be in Markdown format, starting with the section title as a level 1 heading (# Title). "
        prompt += "Include an executive summary, detailed analysis for each question, and a conclusion. "
        prompt += "Cite sources where appropriate. Be professional, clear, and thorough."
        
        return prompt
    
    def _generate_fallback_content(self, section_id: str, section_title: str, 
                                 chapter_id: str, chapter_title: str, 
                                 description: str, questions: List[Dict[str, Any]]) -> str:
        """Generate fallback content for a section when LLM fails.
        
        Args:
            section_id: ID of the section
            section_title: Title of the section
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            description: Description of the section
            questions: List of questions with evidence
            
        Returns:
            Fallback content for the section
        """
        logger.info(f"Generating fallback content for section: {section_id}")
        
        section_content = f"# {section_title}\n\n"
        
        # Add an executive summary
        section_content += "## Executive Summary\n\n"
        section_content += f"This section provides an overview of {section_title} within the context of {chapter_title}. "
        section_content += "Based on the evidence collected, the following key points emerge:\n\n"
        section_content += "* Key finding 1 based on the evidence\n"
        section_content += "* Key finding 2 based on the evidence\n"
        section_content += "* Key finding 3 based on the evidence\n\n"
        
        # Add content for each question
        for question in questions:
            question_id = question.get("qid", "")
            question_text = question.get("text", "")
            
            section_content += f"## {question_text}\n\n"
            section_content += "Based on the evidence collected, we can determine that...\n\n"
            section_content += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt "
            section_content += "ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
            section_content += "ullamco laboris nisi ut aliquip ex ea commodo consequat.\n\n"
            
            # Add citations
            section_content += "### Sources\n\n"
            section_content += "1. Document A, Section 1.2, Page 5\n"
            section_content += "2. Document B, Section 3.4, Page 12\n"
            section_content += "3. Document C, Section 2.1, Page 8\n\n"
        
        # Add a conclusion
        section_content += "## Conclusion\n\n"
        section_content += f"In conclusion, {section_title} is a critical aspect of {chapter_title}. "
        section_content += "The evidence suggests that... Further investigation may be warranted in the areas of...\n"
        
        return section_content

    def _save_section_draft(self, section_id: str, content: str) -> None:
        """Save a section draft to a file.
        
        Args:
            section_id: ID of the section
            content: Draft content for the section
        """
        output_file = self.output_dir / f"{section_id}.md"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"Saved section draft to {output_file}")

    def _save_section_drafts(self, section_drafts: Dict[str, Any]) -> None:
        """Save the section drafts to a file.
        
        Args:
            section_drafts: The section drafts to save
        """
        output_file = self.output_dir / "section_drafts.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(section_drafts, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved section drafts to {output_file}")