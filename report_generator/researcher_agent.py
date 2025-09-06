import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# Import the search function from the RAG pipeline
try:
    import sys
    # Add the parent directory to path to access rag_researcher
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    # Now import from the rag_researcher directory
    from rag_researcher.pipeline_api import search_context
except ImportError as e:
    logging.error(f"Failed to import search_context from rag_researcher.pipeline_api: {e}")
    raise

class ResearcherAgent:
    """Agent that enriches planner output with research data from RAG pipeline."""
    
    def __init__(self, planner_output_dir: str, output_dir: str, top_k: int = 5):
        """
        Initialize the ResearcherAgent.
        
        Args:
            planner_output_dir: Directory containing planner output files
            output_dir: Base output directory for researcher results
            top_k: Number of top results to retrieve for each query
        """
        self.planner_output_dir = Path(planner_output_dir)
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.researcher_output_dir = self.output_dir / "researcher"
        self.sections_output_dir = self.researcher_output_dir / "sections"
        self.sections_output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_planner_sections(self) -> List[Dict[str, Any]]:
        """
        Load all section JSON files from planner output.
        
        Returns:
            List of section dictionaries
        """
        sections_dir = self.planner_output_dir / "sections"
        if not sections_dir.exists():
            raise FileNotFoundError(f"Planner sections directory not found: {sections_dir}")
        
        sections = []
        for json_file in sections_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    section_data = json.load(f)
                    sections.append(section_data)
                    self.logger.info(f"Loaded section: {section_data.get('section_title', 'Unknown')}")
            except Exception as e:
                self.logger.error(f"Failed to load section file {json_file}: {e}")
                continue
        
        return sections
    
    def enrich_section_with_research(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a section with research data by running queries through RAG pipeline.
        
        Args:
            section: Section dictionary from planner output
            
        Returns:
            Enriched section dictionary with research results
        """
        self.logger.info(f"Processing section: {section.get('section_title', 'Unknown')}")
        
        # Extract required fields
        enriched_section = {
            "chapter_title": section.get("chapter_title", ""),
            "section_title": section.get("section_title", ""),
            "description": section.get("description", ""),
            "questions": []
        }
        
        # Process each question
        for question_data in section.get("questions", []):
            enriched_question = {
                "question": question_data.get("question", ""),
                "queries": []
            }
            
            # Process each query in the question
            for query in question_data.get("queries", []):
                self.logger.info(f"Searching for query: {query}")
                
                try:
                    # Run search through RAG pipeline
                    search_results = search_context(query, top_k=self.top_k)
                    
                    # Format results
                    formatted_results = []
                    for result in search_results:
                        formatted_result = {
                            "text": result.get("text", ""),
                            "source": result.get("source", ""),
                            "score": result.get("score", 0.0)
                        }
                        formatted_results.append(formatted_result)
                    
                    enriched_query = {
                        "query": query,
                        "results": formatted_results
                    }
                    
                    enriched_question["queries"].append(enriched_query)
                    self.logger.info(f"Retrieved {len(formatted_results)} results for query: {query}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to search for query '{query}': {e}")
                    # Add empty results on failure
                    enriched_query = {
                        "query": query,
                        "results": []
                    }
                    enriched_question["queries"].append(enriched_query)
            
            enriched_section["questions"].append(enriched_question)
        
        return enriched_section
    
    def save_section_research(self, section: Dict[str, Any], section_id: str):
        """
        Save enriched section research to JSON file.
        
        Args:
            section: Enriched section dictionary
            section_id: Section identifier for filename
        """
        output_file = self.sections_output_dir / f"{section_id}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(section, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved section research to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save section research to {output_file}: {e}")
            raise
    
    def generate_global_research(self, enriched_sections: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """
        Generate global aggregated research JSON with proper hierarchy.
        
        Args:
            enriched_sections: List of enriched section dictionaries
            topic: Research topic
            
        Returns:
            Global research dictionary
        """
        # Group sections by chapter
        chapters_dict = {}
        
        for section in enriched_sections:
            chapter_title = section["chapter_title"]
            
            if chapter_title not in chapters_dict:
                chapters_dict[chapter_title] = {
                    "chapter_title": chapter_title,
                    "sections": []
                }
            
            chapters_dict[chapter_title]["sections"].append(section)
        
        # Convert to list format
        chapters_list = list(chapters_dict.values())
        
        global_research = {
            "topic": topic,
            "chapters": chapters_list
        }
        
        return global_research
    
    def save_global_research(self, global_research: Dict[str, Any]):
        """
        Save global aggregated research to JSON file.
        
        Args:
            global_research: Global research dictionary
        """
        output_file = self.researcher_output_dir / "research.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(global_research, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved global research to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save global research to {output_file}: {e}")
            raise
    
    def run_research(self):
        """
        Main method to run the complete research process.
        """
        self.logger.info("Starting research process...")
        
        # Load planner sections
        sections = self.load_planner_sections()
        if not sections:
            self.logger.warning("No sections found in planner output")
            return
        
        self.logger.info(f"Found {len(sections)} sections to process")
        
        # For testing, only process the first section
        if len(sections) > 1:
            sections = sections[:1]
        self.logger.info(f"Testing mode: Processing only first section: {sections[0].get('section_title', 'Unknown')}")
        
        # Load global plan to get topic
        global_plan_file = self.planner_output_dir / "plan.json"
        topic = "Unknown Topic"
        if global_plan_file.exists():
            try:
                with open(global_plan_file, 'r', encoding='utf-8') as f:
                    global_plan = json.load(f)
                    topic = global_plan.get("topic", "Unknown Topic")
            except Exception as e:
                self.logger.error(f"Failed to load global plan: {e}")
        
        # Process each section
        enriched_sections = []
        for section in sections:
            try:
                # Enrich section with research
                enriched_section = self.enrich_section_with_research(section)
                enriched_sections.append(enriched_section)
                
                # Save individual section
                section_id = section.get("section_id", "unknown")
                self.save_section_research(enriched_section, section_id)
                
            except Exception as e:
                self.logger.error(f"Failed to process section {section.get('section_title', 'Unknown')}: {e}")
                continue
        
        # Generate and save global research
        if enriched_sections:
            global_research = self.generate_global_research(enriched_sections, topic)
            self.save_global_research(global_research)
            
            self.logger.info(f"Research process completed successfully!")
            self.logger.info(f"Processed {len(enriched_sections)} sections")
            self.logger.info(f"Output files:")
            self.logger.info(f"  - Global research: {self.researcher_output_dir / 'research.json'}")
            self.logger.info(f"  - Section research: {self.sections_output_dir}/*.json")
        else:
            self.logger.warning("No sections were successfully processed")