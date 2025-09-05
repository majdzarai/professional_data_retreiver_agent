#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Planner Agent for the Report Generator

This module contains the Planner Agent class that is responsible for generating
questions and RAG queries for each section of a report.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.config import ReportGeneratorConfig
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Planner Agent for the Report Generator.
    
    The Planner Agent is responsible for generating questions and RAG queries
    for each section of a report. It takes a report structure as input and
    produces a plan of questions and queries for each section.
    """

    def __init__(self, config: ReportGeneratorConfig):
        """Initialize the Planner Agent.
        
        Args:
            config: Configuration for the Report Generator
        """
        self.config = config
        self.prompt_template = config.planner_prompt_template
        self.output_dir = Path(config.output_dir) / "planner"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client
        self.llm_client = LLMClient(model=config.model)
        
        logger.info(f"Initialized Planner Agent with output directory: {self.output_dir}")

    def generate_plan(self, report_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan for the report.
        
        Args:
            report_structure: Structure of the report with chapters and sections
            
        Returns:
            A dictionary containing the plan for each section of the report
        """
        logger.info("Generating plan for report")
        
        # Validate report structure
        self._validate_report_structure(report_structure)
        
        # Generate plan for each section
        plan = {
            "report_title": report_structure.get("title", "Untitled Report"),
            "sections": []
        }
        
        # Process each chapter and its sections
        for chapter in report_structure.get("chapters", []):
            chapter_id = chapter.get("id", "")
            chapter_title = chapter.get("title", "")
            
            for section in chapter.get("sections", []):
                section_id = section.get("id", "")
                if not section_id:
                    section_id = f"{chapter_id}_{len(plan['sections'])}"
                    
                section_title = section.get("title", "")
                section_description = section.get("description", "")
                
                # Generate questions and queries for this section
                section_plan = self._generate_section_plan(
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    section_id=section_id,
                    section_title=section_title,
                    section_description=section_description
                )
                
                plan["sections"].append(section_plan)
        
        # Save the plan to a file
        self._save_plan(plan)
        
        return plan

    def _validate_report_structure(self, report_structure: Dict[str, Any]) -> None:
        """Validate the report structure.
        
        Args:
            report_structure: Structure of the report with chapters and sections,
                             or an individual section/chapter
            
        Raises:
            ValueError: If the report structure is invalid
        """
        if not isinstance(report_structure, dict):
            raise ValueError("Report structure must be a dictionary")
        
        # Check if this is a complete report structure or an individual section/chapter
        if "id" in report_structure and "title" in report_structure:
            # This is an individual section or chapter, validate it
            if not isinstance(report_structure["id"], str):
                raise ValueError("Section/chapter 'id' must be a string")
                
            if not isinstance(report_structure["title"], str):
                raise ValueError("Section/chapter 'title' must be a string")
            
            # If it has sections, validate them
            if "sections" in report_structure:
                if not isinstance(report_structure["sections"], list):
                    raise ValueError("'sections' must be a list")
                
                for j, section in enumerate(report_structure["sections"]):
                    if not isinstance(section, dict):
                        raise ValueError(f"Section {j} must be a dictionary")
                        
                    if "title" not in section:
                        raise ValueError(f"Section {j} must have a 'title'")
        else:
            # This is a complete report structure, validate it
            if "chapters" not in report_structure:
                raise ValueError("Report structure must contain 'chapters' key")
                
            if not isinstance(report_structure["chapters"], list):
                raise ValueError("'chapters' must be a list")
                
            for i, chapter in enumerate(report_structure["chapters"]):
                if not isinstance(chapter, dict):
                    raise ValueError(f"Chapter {i} must be a dictionary")
                    
                if "id" not in chapter:
                    raise ValueError(f"Chapter {i} must have an 'id'")
                    
                if "title" not in chapter:
                    raise ValueError(f"Chapter {i} must have a 'title'")
                
                # If it has sections, validate them
                if "sections" in chapter:
                    if not isinstance(chapter["sections"], list):
                        raise ValueError(f"'sections' in chapter {i} must be a list")
                    
                    for j, section in enumerate(chapter["sections"]):
                        if not isinstance(section, dict):
                            raise ValueError(f"Section {j} in chapter {i} must be a dictionary")
                            
                        if "title" not in section:
                            raise ValueError(f"Section {j} in chapter {i} must have a 'title'")

    def _generate_section_plan(
        self,
        chapter_id: str,
        chapter_title: str,
        section_id: str,
        section_title: str,
        section_description: str
    ) -> Dict[str, Any]:
        """Generate a plan for a section.
        
        Args:
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            section_id: ID of the section
            section_title: Title of the section
            section_description: Description of the section
            
        Returns:
            A dictionary containing the plan for the section
        """
        logger.info(f"Generating plan for section: {section_id} - {section_title}")
        
        # Prepare the prompt for the LLM
        prompt = self._prepare_section_prompt(
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            section_id=section_id,
            section_title=section_title,
            section_description=section_description
        )
        
        try:
            # Call the LLM to generate the plan
            logger.info("Calling LLM to generate section plan")
            llm_response = self.llm_client.generate(prompt)
            
            # Parse the LLM response as JSON
            try:
                # Try to extract JSON from the response
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}')
                
                if json_start >= 0 and json_end >= 0:
                    json_str = llm_response[json_start:json_end+1]
                    section_plan = json.loads(json_str)
                else:
                    # If no JSON is found, use the fallback plan
                    logger.warning("No JSON found in LLM response, using fallback plan")
                    section_plan = self._generate_fallback_plan(
                        chapter_id, chapter_title, section_id, section_title, section_description
                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON, using fallback plan")
                section_plan = self._generate_fallback_plan(
                    chapter_id, chapter_title, section_id, section_title, section_description
                )
                
            # Ensure the section plan has the required fields
            section_plan = self._validate_and_fix_section_plan(
                section_plan, chapter_id, chapter_title, section_id, section_title, section_description
            )
            
            return section_plan
            
        except Exception as e:
            logger.error(f"Error generating section plan: {e}")
            # Use fallback plan in case of error
            return self._generate_fallback_plan(
                chapter_id, chapter_title, section_id, section_title, section_description
            )
    
    def _prepare_section_prompt(self, chapter_id: str, chapter_title: str, 
                               section_id: str, section_title: str, 
                               section_description: str) -> str:
        """Prepare the prompt for generating a section plan.
        
        Args:
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            section_id: ID of the section
            section_title: Title of the section
            section_description: Description of the section
            
        Returns:
            The prompt for the LLM
        """
        # Start with the base prompt template
        prompt = self.prompt_template
        
        # Add section-specific information
        prompt += f"\n\nYou are planning the '{section_title}' section of the '{chapter_title}' chapter."
        
        if section_description:
            prompt += f"\n\nSection description: {section_description}"
        
        prompt += f"\n\nPlease generate a plan for this section with the following structure:\n"
        prompt += "{"
        prompt += f"\n  \"section_id\": \"{section_id}\","
        prompt += f"\n  \"section_title\": \"{section_title}\","
        prompt += f"\n  \"chapter_id\": \"{chapter_id}\","
        prompt += f"\n  \"chapter_title\": \"{chapter_title}\","
        prompt += f"\n  \"description\": \"{section_description}\","
        prompt += "\n  \"questions\": ["
        prompt += "\n    {\n      \"qid\": \"unique_question_id\",\n      \"text\": \"clear professional question\",\n      \"queries\": [\"query 1\", \"query 2\", \"query 3\"]\n    },"
        prompt += "\n    ...\n  ]\n}"
        
        return prompt
    
    def _generate_fallback_plan(self, chapter_id: str, chapter_title: str, 
                              section_id: str, section_title: str, 
                              section_description: str) -> Dict[str, Any]:
        """Generate a fallback plan for a section when LLM fails.
        
        Args:
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            section_id: ID of the section
            section_title: Title of the section
            section_description: Description of the section
            
        Returns:
            A dictionary containing the fallback plan for the section
        """
        logger.info(f"Generating fallback plan for section: {section_id}")
        
        section_plan = {
            "section_id": section_id,
            "section_title": section_title,
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "description": section_description,
            "questions": [
                {
                    "qid": f"{section_id}_q1",
                    "text": f"What are the key aspects of {section_title}?",
                    "queries": [
                        f"key aspects of {section_title} in {chapter_title}",
                        f"important elements of {section_title}",
                        f"main components of {section_title} in financial reporting"
                    ]
                },
                {
                    "qid": f"{section_id}_q2",
                    "text": f"What are the regulatory requirements for {section_title}?",
                    "queries": [
                        f"regulatory requirements for {section_title}",
                        f"compliance standards for {section_title} in banking",
                        f"legal framework for {section_title} in financial institutions"
                    ]
                },
                {
                    "qid": f"{section_id}_q3",
                    "text": f"What are the best practices for {section_title}?",
                    "queries": [
                        f"best practices for {section_title} in banking industry",
                        f"industry standards for {section_title}",
                        f"leading approaches to {section_title} in financial sector"
                    ]
                }
            ]
        }
        
        return section_plan
    
    def _validate_and_fix_section_plan(self, section_plan: Dict[str, Any], 
                                     chapter_id: str, chapter_title: str,
                                     section_id: str, section_title: str,
                                     section_description: str) -> Dict[str, Any]:
        """Validate and fix a section plan to ensure it has all required fields.
        
        Args:
            section_plan: The section plan to validate
            chapter_id: ID of the chapter
            chapter_title: Title of the chapter
            section_id: ID of the section
            section_title: Title of the section
            section_description: Description of the section
            
        Returns:
            A validated and fixed section plan
        """
        # Ensure the section plan is a dictionary
        if not isinstance(section_plan, dict):
            logger.warning("Section plan is not a dictionary, using fallback plan")
            return self._generate_fallback_plan(
                chapter_id, chapter_title, section_id, section_title, section_description
            )
        
        # Ensure the section plan has the required fields
        if "section_id" not in section_plan:
            section_plan["section_id"] = section_id
        
        if "section_title" not in section_plan:
            section_plan["section_title"] = section_title
        
        if "chapter_id" not in section_plan:
            section_plan["chapter_id"] = chapter_id
        
        if "chapter_title" not in section_plan:
            section_plan["chapter_title"] = chapter_title
        
        if "description" not in section_plan:
            section_plan["description"] = section_description
        
        if "questions" not in section_plan or not isinstance(section_plan["questions"], list):
            logger.warning("Section plan has no questions or questions is not a list, using fallback questions")
            section_plan["questions"] = self._generate_fallback_plan(
                chapter_id, chapter_title, section_id, section_title, section_description
            )["questions"]
        
        # Validate each question
        for i, question in enumerate(section_plan["questions"]):
            if not isinstance(question, dict):
                logger.warning(f"Question {i} is not a dictionary, replacing with fallback question")
                section_plan["questions"][i] = self._generate_fallback_plan(
                    chapter_id, chapter_title, section_id, section_title, section_description
                )["questions"][0]
                continue
            
            if "qid" not in question:
                question["qid"] = f"{section_id}_q{i+1}"
            
            if "text" not in question:
                question["text"] = f"What are the key aspects of {section_title}?"
            
            if "queries" not in question or not isinstance(question["queries"], list):
                question["queries"] = [
                    f"key aspects of {section_title} in {chapter_title}",
                    f"important elements of {section_title}",
                    f"main components of {section_title} in financial reporting"
                ]
        
        return section_plan

    def _save_plan(self, plan: Dict[str, Any]) -> None:
        """Save the plan to a file.
        
        Args:
            plan: The plan to save
        """
        output_file = self.output_dir / "plan.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved plan to {output_file}")