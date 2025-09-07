import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from llm_client import LLMClient

class WriterAgent:
    """
    Writer Agent that processes section prompts and generates actual content for each section.
    Takes structured prompts from the Section Report Agent and creates well-written report sections.
    """
    
    def __init__(self, output_dir: str = "output/writer"):
        """
        Initialize the Writer Agent.
        
        Args:
            output_dir: Directory to save generated content
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.sections_dir = self.output_dir / "sections"
        self.chapters_dir = self.output_dir / "chapters"
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_client = LLMClient()
        self.logger = logging.getLogger(__name__)
        
        # Track generated content
        self.generated_sections = []
        self.chapters = {}
    
    def process_section_prompts(self, prompts_dir: str) -> bool:
        """
        Process all section prompts in the given directory.
        
        Args:
            prompts_dir: Directory containing section prompt files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            prompts_path = Path(prompts_dir)
            
            # Load prompts summary
            summary_file = prompts_path / "prompts_summary.json"
            if not summary_file.exists():
                self.logger.error(f"Prompts summary not found: {summary_file}")
                return False
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            generated_prompts = summary.get('generated_prompts', {})
            self.logger.info(f"Processing {len(generated_prompts)} sections")
            
            # Process each section
            for section_key, section_info in generated_prompts.items():
                prompt_file_path = section_info['prompt_file']
                # Convert to relative path if it's absolute
                if Path(prompt_file_path).is_absolute():
                    prompt_file = Path(prompt_file_path)
                else:
                    prompt_file = prompts_path / "prompts" / Path(prompt_file_path).name
                
                if not prompt_file.exists():
                    self.logger.warning(f"Prompt file not found: {prompt_file}")
                    continue
                
                success = self._process_single_section(prompt_file)
                if not success:
                    self.logger.error(f"Failed to process section: {section_info['section_title']}")
                    return False
            
            # Generate chapter files
            self._generate_chapter_files()
            
            # Generate final report summary
            self._generate_report_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing section prompts: {str(e)}")
            return False
    
    def _process_single_section(self, prompt_file: Path) -> bool:
        """
        Process a single section prompt and generate content.
        
        Args:
            prompt_file: Path to the section prompt JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load prompt data
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            section_metadata = prompt_data['section_metadata']
            writing_instructions = prompt_data['writing_instructions']
            research_findings = prompt_data['research_findings']
            
            self.logger.info(f"Generating content for section: {section_metadata['section_title']}")
            
            # Create the writing prompt for the LLM
            llm_prompt = self._create_llm_prompt(section_metadata, writing_instructions, research_findings)
            
            # Generate content using LLM
            raw_content = self.llm_client.generate(llm_prompt, max_tokens=2000)
            
            if not raw_content:
                self.logger.error(f"Failed to generate content for section: {section_metadata['section_title']}")
                return False
            
            # Extract actual content from LLM response (handle JSON format if present)
            content = self._extract_content_from_response(raw_content)
            
            # Save section content
            section_data = {
                "metadata": section_metadata,
                "content": content,
                "word_count": len(content.split()),
                "generated_at": self._get_timestamp()
            }
            
            # Create safe filename
            safe_title = self._create_safe_filename(section_metadata['section_title'])
            section_file = self.sections_dir / f"{safe_title}.json"
            
            with open(section_file, 'w', encoding='utf-8') as f:
                json.dump(section_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Section content saved to: {section_file}")
            
            # Track for chapter organization
            chapter_title = section_metadata['chapter_title']
            if chapter_title not in self.chapters:
                self.chapters[chapter_title] = []
            
            self.chapters[chapter_title].append({
                "section_title": section_metadata['section_title'],
                "section_file": str(section_file),
                "content": content,
                "word_count": len(content.split())
            })
            
            self.generated_sections.append(section_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing section {prompt_file}: {str(e)}")
            return False
    
    def _create_llm_prompt(self, metadata: Dict, instructions: Dict, research: List[Dict]) -> str:
        """
        Create a comprehensive prompt for the LLM to generate section content.
        
        Args:
            metadata: Section metadata
            instructions: Writing instructions
            research: Research findings
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Extract key research information
        research_summary = self._extract_research_summary(research)
        
        prompt = f"""You are a professional business report writer. Generate a comprehensive and well-structured section for a business report.

**SECTION DETAILS:**
Chapter: {metadata['chapter_title']}
Section Title: {metadata['section_title']}
Description: {metadata['description']}

**WRITING OBJECTIVE:**
{instructions['objective']}

**SCOPE:**
{instructions['scope']}

**STRUCTURE REQUIREMENTS:**
{chr(10).join(f"- {req}" for req in instructions['structure_requirements'])}

**CONTENT GUIDELINES:**
{chr(10).join(f"- {guideline}" for guideline in instructions['content_guidelines'])}

**RESEARCH FINDINGS:**
{research_summary}

**INSTRUCTIONS:**
1. Write a professional, well-structured section that addresses all the requirements above
2. Use the research findings to support your content with specific facts and data
3. Maintain a formal business report tone throughout
4. Ensure the content flows logically and is easy to read
5. Include specific details such as dates, numbers, and official names where available
6. Do not include section headers or titles in your response - just the content
7. Write in paragraph form with proper transitions between ideas

Generate the section content now:"""
        
        return prompt
    
    def _extract_content_from_response(self, raw_content: str) -> str:
        """
        Extract actual content from LLM response, handling JSON format if present.
        
        Args:
            raw_content: Raw response from LLM
            
        Returns:
            str: Cleaned content text
        """
        try:
            # Try to parse as JSON first
            if raw_content.strip().startswith('[') or raw_content.strip().startswith('{'):
                parsed = json.loads(raw_content)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Extract from first item if it's a list
                    if isinstance(parsed[0], dict) and 'question' in parsed[0]:
                        return parsed[0]['question']
                elif isinstance(parsed, dict):
                    # Extract from dict structure
                    if 'question' in parsed:
                        return parsed['question']
                    elif 'content' in parsed:
                        return parsed['content']
            
            # If not JSON or parsing failed, return as-is
            return raw_content.strip()
            
        except (json.JSONDecodeError, KeyError, IndexError):
            # If JSON parsing fails, return the raw content
            return raw_content.strip()
    
    def _extract_research_summary(self, research_findings: List[Dict]) -> str:
        """
        Extract and format key research information for the LLM prompt.
        
        Args:
            research_findings: List of research findings
            
        Returns:
            str: Formatted research summary
        """
        summary_parts = []
        
        for finding in research_findings:
            question = finding['question']
            key_info = finding.get('key_information', [])
            
            if key_info:
                summary_parts.append(f"\n**{question}**")
                for info in key_info[:3]:  # Limit to top 3 pieces of info
                    if info.strip():
                        summary_parts.append(f"- {info.strip()}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific research findings available."
    
    def _generate_chapter_files(self):
        """
        Generate consolidated chapter files containing all sections for each chapter.
        """
        for chapter_title, sections in self.chapters.items():
            chapter_data = {
                "chapter_title": chapter_title,
                "sections": sections,
                "total_sections": len(sections),
                "total_word_count": sum(section['word_count'] for section in sections),
                "generated_at": self._get_timestamp()
            }
            
            # Create chapter content by combining all sections
            chapter_content = f"# {chapter_title}\n\n"
            for section in sections:
                chapter_content += f"## {section['section_title']}\n\n"
                chapter_content += section['content'] + "\n\n"
            
            chapter_data['full_content'] = chapter_content
            
            # Save chapter file
            safe_chapter_name = self._create_safe_filename(chapter_title)
            chapter_file = self.chapters_dir / f"{safe_chapter_name}.json"
            
            with open(chapter_file, 'w', encoding='utf-8') as f:
                json.dump(chapter_data, f, indent=2, ensure_ascii=False)
            
            # Also save as markdown
            md_file = self.chapters_dir / f"{safe_chapter_name}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(chapter_content)
            
            self.logger.info(f"Chapter saved: {chapter_file}")
    
    def _generate_report_summary(self):
        """
        Generate a summary of the entire report generation process.
        """
        summary = {
            "report_generation_summary": {
                "total_chapters": len(self.chapters),
                "total_sections": len(self.generated_sections),
                "total_word_count": sum(section['word_count'] for section in self.generated_sections),
                "chapters": list(self.chapters.keys()),
                "generated_at": self._get_timestamp()
            },
            "chapters_detail": {
                chapter: {
                    "sections_count": len(sections),
                    "word_count": sum(s['word_count'] for s in sections),
                    "sections": [s['section_title'] for s in sections]
                }
                for chapter, sections in self.chapters.items()
            },
            "output_files": {
                "sections_directory": str(self.sections_dir),
                "chapters_directory": str(self.chapters_dir)
            }
        }
        
        summary_file = self.output_dir / "report_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Report summary saved: {summary_file}")
    
    def _create_safe_filename(self, title: str) -> str:
        """
        Create a safe filename from a title.
        
        Args:
            title: Original title
            
        Returns:
            str: Safe filename
        """
        # Replace spaces and special characters
        safe_name = title.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        # Remove other problematic characters
        safe_chars = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
        return safe_chars[:50]  # Limit length
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp as string.
        
        Returns:
            str: Current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()