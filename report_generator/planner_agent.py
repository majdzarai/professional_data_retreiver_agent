import json
import os
from pathlib import Path
import logging
from llm_client import LLMClient
import re
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlannerAgent:
    """Agent that generates a plan for each section of a report structure."""
    
    def __init__(self, structure_path):
        """Initialize the planner agent with the path to the report structure.
        
        Args:
            structure_path (str): Path to the report structure JSON file.
        """
        self.structure_path = structure_path
        self.structure = self._load_structure()
        self.output_dir = Path("output/planner")
        self.sections_dir = self.output_dir / "sections"
        
        # Initialize LLM client
        self.llm_client = LLMClient(model="llama3.1")
        
        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_structure(self):
        """Load the report structure from the JSON file.
        
        Returns:
            dict: The report structure.
        """
        try:
            with open(self.structure_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading report structure: {e}")
            raise
    
    def generate_plan(self, topic):
        """Generate a plan for each section of the report structure.
        
        Args:
            topic (str): The topic of the report.
            
        Returns:
            dict: The complete plan.
        """
        logger.info(f"Generating plan for topic: {topic}")
        
        # Initialize the global plan
        global_plan = {
            "topic": topic,
            "report_title": self.structure.get("report_title", "Report"),
            "sections": []
        }
        
        # Process only the first section for testing
        sections_processed = 0
        for chapter in self.structure.get("chapters", []):
            if sections_processed >= 1:
                break
                
            chapter_id = chapter.get("id")
            chapter_title = chapter.get("title")
            
            # Check if the chapter has sections
            if "sections" in chapter:
                # Process only the first section in the chapter
                for section in chapter["sections"]:
                    logger.info(f"Testing mode: Processing only first section: {section.get('title')}")
                    section_plan = self._generate_section_plan(chapter_id, chapter_title, section, topic)
                    global_plan["sections"].append({
                        "section_id": section["id"],
                        "section_title": section["title"],
                        "chapter_id": chapter_id,
                        "chapter_title": chapter_title
                    })
                    
                    # Save the section plan to a file
                    self._save_section_plan(section_plan)
                    sections_processed += 1
                    break
            else:
                # The chapter itself is a section
                logger.info(f"Testing mode: Processing only first chapter as section: {chapter_title}")
                section_plan = self._generate_section_plan(chapter_id, chapter_title, chapter, topic)
                global_plan["sections"].append({
                    "section_id": chapter_id,
                    "section_title": chapter_title,
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_title
                })
                
                # Save the section plan to a file
                self._save_section_plan(section_plan)
                sections_processed += 1
        
        # Save the global plan
        self._save_global_plan(global_plan)
        
        return global_plan
    
    def _generate_section_plan(self, chapter_id, chapter_title, section, topic):
        """Generate a plan for a specific section.
        
        Args:
            chapter_id (str): The ID of the chapter.
            chapter_title (str): The title of the chapter.
            section (dict): The section data.
            topic (str): The topic of the report.
            
        Returns:
            dict: The section plan.
        """
        section_id = section.get("id")
        section_title = section.get("title")
        description = section.get("description", "")
        
        logger.info(f"Generating plan for section: {section_title}")
        
        # Generate questions and queries based on the section
        questions = self._generate_questions(section_id, section_title, description, topic)
        
        # Create the section plan
        section_plan = {
            "section_id": section_id,
            "section_title": section_title,
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "description": description,
            "topic": topic,
            "questions": questions
        }
        
        return section_plan

    def extract_and_validate_questions(self, response: str) -> list:
        """
        Extract and validate questions from LLM response.
        
        Args:
            response (str): Raw LLM response text
            
        Returns:
            list: List of validated question dictionaries
        """
        try:
            logger.info(f"Processing LLM response: {response[:200]}...")
            
            # Step 1: Try to find JSON array directly (clean response)
            json_text = None
            
            # Pattern 1: Response starts directly with JSON array
            direct_match = re.search(r'^\s*(\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\])\s*$', response.strip(), re.MULTILINE)
            if direct_match:
                json_text = direct_match.group(1)
                logger.info("Found direct JSON array response")
            
            # Pattern 2: JSON in markdown code blocks
            if not json_text:
                json_match = re.search(r"```(?:json)?\s*\n?(\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\])\s*\n?```", response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    json_text = json_match.group(1)
                    logger.info("Found JSON in markdown code block")
            
            # Pattern 3: Extract JSON from mixed content (with explanatory text)
            if not json_text:
                # Look for JSON arrays in the response, even with surrounding text
                array_patterns = [
                    # Most restrictive: Complete JSON array with question/queries structure
                    r'(\[\s*\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"queries"\s*:\s*\[[^\]]*\]\s*\}(?:\s*,\s*\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"queries"\s*:\s*\[[^\]]*\]\s*\})*\s*\])',
                    # More flexible: JSON array with required fields (multiline)
                    r'(\[\s*\{[^\[\]{}]*"question"[^\[\]{}]*"queries"[^\[\]{}]*\}(?:[^\[\]{}]*\{[^\[\]{}]*"question"[^\[\]{}]*"queries"[^\[\]{}]*\})*[^\[\]{}]*\])',
                    # Most flexible: Any JSON array structure
                    r'(\[\s*\{[\s\S]*?"question"[\s\S]*?"queries"[\s\S]*?\}[\s\S]*?\])'
                ]
                
                json_candidates = []
                for pattern in array_patterns:
                    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
                    for match in matches:
                        if '"question"' in match and '"queries"' in match:
                            json_candidates.append(match.strip())
                
                # Validate candidates and pick the best one
                for candidate in sorted(set(json_candidates), key=len, reverse=True):
                    try:
                        test_data = json.loads(candidate)
                        if isinstance(test_data, list) and len(test_data) > 0:
                            first_item = test_data[0]
                            if (isinstance(first_item, dict) and 
                                'question' in first_item and 
                                'queries' in first_item and
                                isinstance(first_item['question'], str) and
                                isinstance(first_item['queries'], list)):
                                json_text = candidate
                                logger.info(f"Found valid JSON array with {len(test_data)} questions")
                                break
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Candidate validation failed: {e}")
                        continue
                
                if not json_text:
                    raise ValueError("No valid JSON array with proper question/queries structure found")
            
            # Clean up the JSON text
            json_text = json_text.strip()
            
            # Parse JSON
            questions = json.loads(json_text)
            
            if not isinstance(questions, list):
                logger.warning("JSON is not a list, wrapping in array")
                questions = [questions] if isinstance(questions, dict) else []
            
            # Validate and repair questions
            valid_questions = []
            for i, item in enumerate(questions):
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid question at index {i}: not a dictionary")
                    continue
                    
                # Validate question field
                if "question" not in item or not isinstance(item["question"], str) or not item["question"].strip():
                    logger.warning(f"Repairing missing/invalid question at index {i}")
                    item["question"] = "What are the key aspects to consider?"
                else:
                    item["question"] = item["question"].strip()
                    
                # Validate queries field    
                if "queries" not in item or not isinstance(item["queries"], list):
                    logger.warning(f"Repairing missing/invalid queries at index {i}")
                    item["queries"] = [item["question"]]
                else:
                    # Ensure all queries are non-empty strings
                    clean_queries = []
                    for q in item["queries"]:
                        if isinstance(q, str) and q.strip():
                            clean_queries.append(q.strip())
                    
                    if not clean_queries:
                        logger.warning(f"No valid queries found for question {i}, using question as query")
                        item["queries"] = [item["question"]]
                    else:
                        item["queries"] = clean_queries
                        
                valid_questions.append(item)
                
            if not valid_questions:
                logger.warning("No valid questions found, using fallback")
                return [{
                    "question": "What are the key aspects to consider?",
                    "queries": ["key aspects", "important considerations"]
                }]
                
            logger.info(f"Successfully extracted {len(valid_questions)} valid questions")
            return valid_questions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Attempted to parse: {json_text if 'json_text' in locals() else 'N/A'}")
        except Exception as e:
            logger.error(f"Failed to process questions: {e}")
            
        # Fallback for any error
        logger.warning("Using fallback question due to processing error")
        return [{
            "question": "What are the key aspects to consider?",
            "queries": ["key aspects", "important considerations"]
        }]
    
    def _generate_questions(self, section_id, section_title, description, topic):
        """
        Generate questions and queries for a section using the LLM.
        """
        prompt = f"""
        You are a professional research planner for structured reports.

        Task:
        Generate 3-5 clear, precise research questions for the section:
        - Section ID: {section_id}
        - Section Title: {section_title}
        - Section Description: {description}
        - Report Topic: {topic}

        For each question, also generate 2-3 optimized search queries that could be
        used in a retrieval pipeline to collect relevant evidence.

        CRITICAL INSTRUCTIONS:
        - Return ONLY the JSON array, no explanations, no markdown, no additional text
        - Do NOT include phrases like "Here is the output:" or "The JSON array is:"
        - Start your response directly with the opening bracket [
        - End your response directly with the closing bracket ]
        - Use this exact format:
        [
          {{
            "question": "What specific aspect needs to be researched?",
            "queries": ["search term 1", "search term 2", "search term 3"]
          }},
          {{
            "question": "Another research question?",
            "queries": ["query 1", "query 2"]
          }}
        ]
        """

        response = self.llm_client.generate(prompt, max_tokens=800)
        return self.extract_and_validate_questions(response)
    
    def _save_section_plan(self, section_plan):
        """Save a section plan to a JSON file.
        
        Args:
            section_plan (dict): The section plan to save.
        """
        section_id = section_plan["section_id"]
        file_path = self.sections_dir / f"{section_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(section_plan, f, indent=2)
            logger.info(f"Saved section plan to {file_path}")
        except Exception as e:
            logger.error(f"Error saving section plan: {e}")
            raise
    
    def _save_global_plan(self, global_plan):
        """Save the global plan to a JSON file.
        
        Args:
            global_plan (dict): The global plan to save.
        """
        file_path = self.output_dir / "plan.json"
        
        # Create a new structure organized by chapters
        chapters_plan = {}
        
        # Group sections by chapter
        for section in global_plan.get("sections", []):
            chapter_id = section.get("chapter_id")
            chapter_title = section.get("chapter_title")
            section_id = section.get("section_id")
            
            # Initialize chapter if not exists
            if chapter_id not in chapters_plan:
                chapters_plan[chapter_id] = {
                    "title": chapter_title,
                    "sections": {}
                }
            
            # Load the section plan with questions
            section_file = self.sections_dir / f"{section_id}.json"
            try:
                with open(section_file, 'r') as f:
                    section_data = json.load(f)
                    # Add section with its questions to the chapter
                    chapters_plan[chapter_id]["sections"][section_id] = {
                        "title": section_data.get("section_title"),
                        "questions": section_data.get("questions", [])
                    }
            except Exception as e:
                logger.error(f"Error loading section plan {section_id}: {e}")
                # Add empty section if file can't be loaded
                chapters_plan[chapter_id]["sections"][section_id] = {
                    "title": section.get("section_title"),
                    "questions": []
                }
        
        # Create the final output structure
        final_plan = {
            "topic": global_plan.get("topic"),
            "report_title": global_plan.get("report_title"),
            "chapters": chapters_plan
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_plan, f, indent=2)
            logger.info(f"Saved global plan to {file_path}")
        except Exception as e:
            logger.error(f"Error saving global plan: {e}")
            raise