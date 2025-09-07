#!/usr/bin/env python3
"""
Section Report Agent

This agent processes researcher output and generates structured writing prompts
for each section that can be consumed by a writer agent.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SectionReportAgent:
    """
    Agent that processes researcher output and generates structured writing prompts
    for each section of the report.
    """
    
    def __init__(self, output_dir: str = "output/section_prompts"):
        """
        Initialize the Section Report Agent.
        
        Args:
            output_dir (str): Directory to save generated prompts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.prompts_dir = self.output_dir / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        
        logger.info(f"Section Report Agent initialized with output directory: {self.output_dir}")
    
    def process_researcher_output(self, researcher_dir: str) -> Dict[str, Any]:
        """
        Process researcher output and generate writing prompts for each section.
        
        Args:
            researcher_dir (str): Directory containing researcher output
            
        Returns:
            Dict[str, Any]: Summary of generated prompts
        """
        researcher_path = Path(researcher_dir)
        
        # Load main research file
        research_file = researcher_path / "research.json"
        if not research_file.exists():
            raise FileNotFoundError(f"Research file not found: {research_file}")
        
        with open(research_file, 'r', encoding='utf-8') as f:
            research_data = json.load(f)
        
        # Process sections directory
        sections_dir = researcher_path / "sections"
        if not sections_dir.exists():
            raise FileNotFoundError(f"Sections directory not found: {sections_dir}")
        
        generated_prompts = {}
        
        # Process each section file
        for section_file in sections_dir.glob("*.json"):
            section_name = section_file.stem
            logger.info(f"Processing section: {section_name}")
            
            with open(section_file, 'r', encoding='utf-8') as f:
                section_data = json.load(f)
            
            # Generate prompt for this section
            prompt = self._generate_section_prompt(section_data)
            
            # Save prompt to file
            prompt_file = self.prompts_dir / f"{section_name}_prompt.json"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(prompt, f, indent=2, ensure_ascii=False)
            
            generated_prompts[section_name] = {
                "prompt_file": str(prompt_file),
                "section_title": section_data.get("section_title", "Unknown"),
                "chapter_title": section_data.get("chapter_title", "Unknown"),
                "questions_count": len(section_data.get("questions", []))
            }
            
            logger.info(f"Generated prompt for section '{section_name}' saved to: {prompt_file}")
        
        # Save summary
        summary = {
            "topic": research_data.get("topic", "Unknown"),
            "total_sections": len(generated_prompts),
            "generated_prompts": generated_prompts,
            "output_directory": str(self.output_dir)
        }
        
        summary_file = self.output_dir / "prompts_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Section prompts generation completed. Summary saved to: {summary_file}")
        return summary
    
    def _generate_section_prompt(self, section_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured writing prompt for a specific section.
        
        Args:
            section_data (Dict[str, Any]): Section data from researcher output
            
        Returns:
            Dict[str, Any]: Structured writing prompt
        """
        chapter_title = section_data.get("chapter_title", "Unknown Chapter")
        section_title = section_data.get("section_title", "Unknown Section")
        description = section_data.get("description", "No description available")
        questions = section_data.get("questions", [])
        
        # Extract key information from research results
        research_findings = self._extract_research_findings(questions)
        
        # Generate structured prompt
        prompt = {
            "section_metadata": {
                "chapter_title": chapter_title,
                "section_title": section_title,
                "description": description,
                "total_questions": len(questions)
            },
            "writing_instructions": {
                "objective": f"Write a comprehensive and professional section titled '{section_title}' for the chapter '{chapter_title}'",
                "scope": description,
                "structure_requirements": [
                    "Start with a clear introduction to the section topic",
                    "Present information in a logical, well-organized manner",
                    "Use professional business report language and tone",
                    "Include specific data points and facts from the research findings",
                    "Ensure accuracy and cite relevant information appropriately",
                    "Conclude with a summary of key points if the section is substantial"
                ],
                "content_guidelines": [
                    "Focus on factual, objective information",
                    "Use clear, concise language suitable for business stakeholders",
                    "Organize information hierarchically (most important first)",
                    "Include specific details such as dates, numbers, and official names",
                    "Maintain consistency with overall report structure and tone"
                ]
            },
            "research_findings": research_findings,
            "key_questions_addressed": [q.get("question", "Unknown question") for q in questions],
            "writing_prompt": self._create_detailed_writing_prompt(chapter_title, section_title, description, research_findings)
        }
        
        return prompt
    
    def _extract_research_findings(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and organize research findings from questions and their results.
        
        Args:
            questions (List[Dict[str, Any]]): List of questions with research results
            
        Returns:
            List[Dict[str, Any]]: Organized research findings
        """
        findings = []
        
        for question in questions:
            question_text = question.get("question", "Unknown question")
            queries = question.get("queries", [])
            
            # Collect all unique research results
            all_results = []
            seen_texts = set()
            
            for query in queries:
                query_text = query.get("query", "Unknown query")
                results = query.get("results", [])
                
                for result in results:
                    result_text = result.get("text", "").strip()
                    if result_text and result_text not in seen_texts:
                        all_results.append({
                            "text": result_text,
                            "source": result.get("source", "Unknown source"),
                            "score": result.get("score", 0.0),
                            "query": query_text
                        })
                        seen_texts.add(result_text)
            
            if all_results:
                findings.append({
                    "question": question_text,
                    "total_results": len(all_results),
                    "results": all_results[:5],  # Limit to top 5 results per question
                    "key_information": self._extract_key_information(all_results)
                })
        
        return findings
    
    def _extract_key_information(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key information points from research results.
        
        Args:
            results (List[Dict[str, Any]]): Research results
            
        Returns:
            List[str]: Key information points
        """
        key_info = []
        
        for result in results[:3]:  # Focus on top 3 results
            text = result.get("text", "")
            if text:
                # Extract key sentences or phrases (simplified approach)
                sentences = text.split('. ')
                for sentence in sentences[:2]:  # Take first 2 sentences
                    if len(sentence.strip()) > 20:  # Filter out very short fragments
                        key_info.append(sentence.strip())
        
        return key_info[:5]  # Limit to 5 key points
    
    def _create_detailed_writing_prompt(self, chapter_title: str, section_title: str, 
                                      description: str, research_findings: List[Dict[str, Any]]) -> str:
        """
        Create a detailed writing prompt for the writer agent.
        
        Args:
            chapter_title (str): Chapter title
            section_title (str): Section title
            description (str): Section description
            research_findings (List[Dict[str, Any]]): Research findings
            
        Returns:
            str: Detailed writing prompt
        """
        prompt = f"""# Writing Instructions for Report Section

## Section Details
- **Chapter**: {chapter_title}
- **Section**: {section_title}
- **Purpose**: {description}

## Writing Task
Write a comprehensive, professional section for a business report that addresses the following requirements:

### Content Requirements:
1. **Introduction**: Begin with a clear introduction that establishes the purpose and scope of this section
2. **Main Content**: Present the information in a logical, well-structured manner using the research findings provided
3. **Professional Tone**: Use formal business language appropriate for stakeholders and decision-makers
4. **Factual Accuracy**: Base all statements on the provided research findings and cite specific data points
5. **Clarity**: Ensure the content is clear, concise, and easy to understand

### Structure Guidelines:
- Use appropriate headings and subheadings if the content is substantial
- Present information in order of importance
- Include specific details such as dates, numbers, company names, and official designations
- Maintain consistency with professional report formatting standards

### Research Findings to Incorporate:
"""
        
        # Add research findings
        for i, finding in enumerate(research_findings, 1):
            prompt += f"\n#### Question {i}: {finding['question']}\n"
            
            if finding.get('key_information'):
                prompt += "**Key Information to Include:**\n"
                for info in finding['key_information']:
                    prompt += f"- {info}\n"
            
            if finding.get('results'):
                prompt += "\n**Supporting Research Data:**\n"
                for j, result in enumerate(finding['results'][:2], 1):  # Limit to top 2 results
                    prompt += f"{j}. {result['text'][:200]}...\n\n"
        
        prompt += """\n### Final Instructions:
- Synthesize the research findings into coherent, flowing prose
- Ensure all factual claims are supported by the provided research data
- Maintain professional objectivity throughout the section
- Aim for completeness while being concise
- Review for accuracy, clarity, and professional presentation

**Expected Output**: A well-written, professional report section that fully addresses the section requirements using the provided research findings."""
        
        return prompt