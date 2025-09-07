# Report Generator System

A comprehensive multi-agent system for generating structured reports from research data. The system consists of 5 main agents that work together in a pipeline to create professional PDF reports.

## System Overview

The Report Generator follows a sequential pipeline:

```
Planner → Researcher → Section Report → Writer → PDF Generator
```

Each agent has specific responsibilities and produces outputs that feed into the next stage.

## Agent Details

### 1. Planner Agent

**File:** `planner_agent.py`
**Main Script:** `main_planner.py`

#### What it does:
- Creates the overall structure and plan for the report
- Defines chapters, sections, and their relationships
- Sets the research scope and objectives

#### Input:
- User requirements or topic description
- Optional: existing report structure template

#### Process:
1. Analyzes the research topic
2. Creates a hierarchical report structure
3. Defines research questions for each section
4. Generates a comprehensive plan JSON file

#### Output:
- `plan.json`: Complete report structure with chapters and sections
- `sections/`: Individual section files with research objectives

#### Key Configuration:
```python
# In planner_agent.py
class PlannerAgent:
    def __init__(self, output_dir: str = "output/planner"):
        # Modify output directory here
        
    def create_report_plan(self, topic: str, requirements: str = None):
        # Modify planning logic here
```

#### How to Modify:
- **Change report structure:** Edit the `_create_report_structure()` method
- **Add new section types:** Modify `_generate_sections()` method
- **Customize planning prompts:** Update the LLM prompts in the planning methods

---

### 2. Researcher Agent

**File:** `researcher_agent.py`
**Main Script:** `main_researcher.py`

#### What it does:
- Conducts research for each planned section
- Gathers relevant information from data sources
- Provides evidence and supporting materials

#### Input:
- Plan from Planner Agent (`plan.json`)
- Research data sources (documents, databases)
- Section-specific research questions

#### Process:
1. Reads the report plan and section requirements
2. For each section, conducts targeted research
3. Retrieves relevant information using RAG (Retrieval-Augmented Generation)
4. Compiles research findings with source citations

#### Output:
- `researcher/`: Directory containing research results
- Section-specific research files with findings and sources
- Compiled evidence for each report section

#### Key Configuration:
```python
# In researcher_agent.py
class ResearcherAgent:
    def __init__(self, output_dir: str = "output/researcher"):
        # Modify output directory
        
    def research_sections(self, plan_path: str, data_sources: List[str]):
        # Modify research methodology here
```

#### How to Modify:
- **Change research methods:** Edit `_conduct_section_research()` method
- **Add new data sources:** Modify `_load_data_sources()` method
- **Customize search queries:** Update query generation in research methods
- **Adjust retrieval parameters:** Modify RAG configuration

---

### 3. Section Report Agent

**File:** `section_report_agent.py`
**Main Script:** `main_section_report.py`

#### What it does:
The Section Report Agent is a critical bridge between the Researcher Agent and Writer Agent. It transforms raw research findings into structured, actionable writing prompts that guide the Writer Agent in creating professional report sections.

#### Input:
- **Primary Input:** `researcher_dir` - Directory containing researcher output
  - `research.json` - Main research metadata file
  - `sections/` - Directory with individual section JSON files
  - Each section file contains: chapter_title, section_title, description, questions with research results

#### Detailed Process Flow:

**Step 1: Initialization (`__init__`)**
- **Input:** `output_dir` (default: "output/section_prompts")
- **Process:** Creates output directory structure with `prompts/` subdirectory
- **Output:** Initialized agent with logging setup

**Step 2: Main Processing (`process_researcher_output`)**
- **Input:** Path to researcher output directory
- **Process:**
  1. Loads `research.json` for metadata
  2. Scans `sections/` directory for individual section files
  3. Processes each section file sequentially
  4. Generates structured prompts for each section
  5. Saves individual prompt files and summary
- **Output:** Dictionary with processing summary

**Step 3: Section Prompt Generation (`_generate_section_prompt`)**
- **Input:** Individual section data (JSON object)
- **Process:**
  1. Extracts section metadata (chapter_title, section_title, description)
  2. Processes research questions and findings
  3. Creates structured writing instructions
  4. Generates detailed writing prompt
- **Output:** Comprehensive prompt object with 5 main components:
  - `section_metadata`: Basic section information
  - `writing_instructions`: Detailed writing guidelines
  - `research_findings`: Processed research data
  - `key_questions_addressed`: List of research questions
  - `writing_prompt`: Complete prompt text for Writer Agent

**Step 4: Research Findings Extraction (`_extract_research_findings`)**
- **Input:** List of questions with research results
- **Process:**
  1. Iterates through each research question
  2. Collects unique research results from multiple queries
  3. Deduplicates results based on text content
  4. Limits to top 5 results per question
  5. Extracts key information points
- **Output:** Structured findings with question context and supporting data

**Step 5: Key Information Extraction (`_extract_key_information`)**
- **Input:** Research results array
- **Process:**
  1. Focuses on top 3 research results
  2. Splits text into sentences
  3. Filters sentences by length (>20 characters)
  4. Limits to 5 key information points
- **Output:** Array of key information strings

**Step 6: Detailed Writing Prompt Creation (`_create_detailed_writing_prompt`)**
- **Input:** Chapter title, section title, description, research findings
- **Process:**
  1. Creates comprehensive writing instructions
  2. Includes section context and requirements
  3. Incorporates research findings with specific data
  4. Provides structure guidelines and formatting requirements
- **Output:** Complete text prompt ready for Writer Agent

#### Output Structure:

**File Outputs:**
- `prompts/{section_name}_prompt.json` - Individual section prompts
- `prompts_summary.json` - Complete processing summary

**Prompt JSON Structure:**
```json
{
  "section_metadata": {
    "chapter_title": "string",
    "section_title": "string",
    "description": "string",
    "total_questions": "number"
  },
  "writing_instructions": {
    "objective": "string",
    "scope": "string",
    "structure_requirements": ["array of requirements"],
    "content_guidelines": ["array of guidelines"]
  },
  "research_findings": [
    {
      "question": "string",
      "total_results": "number",
      "results": ["array of research results"],
      "key_information": ["array of key points"]
    }
  ],
  "key_questions_addressed": ["array of questions"],
  "writing_prompt": "complete text prompt"
}
```

#### Key Configuration:
```python
# In section_report_agent.py
class SectionReportAgent:
    def __init__(self, output_dir: str = "output/section_prompts"):
        # Modify output directory structure
        self.output_dir = Path(output_dir)
        self.prompts_dir = self.output_dir / "prompts"
        
    def process_researcher_output(self, researcher_dir: str):
        # Main processing pipeline - modify workflow here
        
    def _generate_section_prompt(self, section_data: Dict[str, Any]):
        # Core prompt generation - modify prompt structure here
        
    def _create_detailed_writing_prompt(self, chapter_title, section_title, description, research_findings):
        # Detailed prompt text - modify writing instructions here
```

#### How to Modify:
- **Change output structure:** Modify `__init__` method directory setup
- **Customize prompt format:** Edit `_generate_section_prompt()` method
- **Modify research processing:** Update `_extract_research_findings()` logic
- **Change writing instructions:** Edit `_create_detailed_writing_prompt()` template
- **Add new metadata:** Extend section_metadata structure
- **Customize filtering:** Modify result limits and filtering criteria

#### Error Handling:
- Validates existence of required input files (`research.json`, `sections/` directory)
- Handles missing or malformed section data gracefully
- Provides detailed logging for debugging and monitoring
- Creates output directories automatically if they don't exist

---

### 4. Writer Agent

**File:** `writer_agent.py`
**Main Script:** `main_writer.py`

#### Role in Pipeline:
The Writer Agent is the content generation engine that transforms structured prompts from the Section Report Agent into professional, well-written report sections. It serves as the bridge between research data and final report content, utilizing Large Language Models (LLMs) to produce high-quality business writing.

#### Input Requirements:

**Primary Input:**
- **Prompts Directory:** Contains section prompt JSON files from Section Report Agent
- **Prompts Summary:** `prompts_summary.json` with processing metadata

**Input Data Structure:**
```json
{
  "section_metadata": {
    "chapter_title": "Market Analysis",
    "section_title": "Competitive Landscape",
    "description": "Analysis of key competitors",
    "total_questions": 5
  },
  "writing_instructions": {
    "objective": "Analyze competitive positioning",
    "scope": "Focus on top 5 competitors",
    "structure_requirements": ["Introduction", "Analysis", "Implications"],
    "content_guidelines": ["Use data-driven insights", "Professional tone"]
  },
  "research_findings": [
    {
      "question": "Who are the main competitors?",
      "key_information": ["Company A: 25% market share", "Company B: 18% market share"]
    }
  ],
  "writing_prompt": "Complete LLM prompt text"
}
```

#### Detailed Process Flow:

**Step 1: Initialization and Setup (`__init__`)**
- **Purpose:** Initialize Writer Agent with output directories and LLM client
- **Process:**
  1. Creates output directory structure (`sections/`, `chapters/`)
  2. Initializes LLM client for content generation
  3. Sets up logging and tracking variables
  4. Prepares chapter organization system
- **Output:** Configured WriterAgent instance ready for processing

**Step 2: Main Processing Pipeline (`process_section_prompts`)**
- **Input:** Directory path containing section prompt files
- **Process:**
  1. Loads and validates prompts summary file
  2. Iterates through each section prompt file
  3. Calls `_process_single_section` for each prompt
  4. Tracks successful and failed processing
  5. Generates consolidated chapter files
  6. Creates comprehensive report summary
- **Output:** Boolean success status and generated content files

**Step 3: Individual Section Processing (`_process_single_section`)**
- **Input:** Single section prompt JSON file
- **Process:**
  1. **Load Prompt Data:** Parses JSON file and extracts components
  2. **Create LLM Prompt:** Formats comprehensive prompt using `_create_llm_prompt`
  3. **Generate Content:** Sends prompt to LLM and receives response
  4. **Extract Content:** Processes LLM response using `_extract_content_from_response`
  5. **Save Section Data:** Creates section JSON file with metadata and content
  6. **Track Chapter Organization:** Organizes sections by chapter for consolidation
- **Output:** Individual section JSON file and chapter tracking data

**Step 4: LLM Prompt Creation (`_create_llm_prompt`)**
- **Input:** Section metadata, writing instructions, research findings
- **Process:**
  1. Extracts research summary using `_extract_research_summary`
  2. Formats comprehensive prompt with:
     - Section details (chapter, title, description)
     - Writing objectives and scope
     - Structure requirements and content guidelines
     - Research findings with specific data points
     - Detailed writing instructions and formatting rules
  3. Creates professional business writing context
- **Output:** Complete formatted prompt string for LLM

**Step 5: Content Extraction (`_extract_content_from_response`)**
- **Input:** Raw LLM response text
- **Process:**
  1. Attempts JSON parsing if response appears to be JSON format
  2. Extracts content from structured response (handles 'question', 'content' fields)
  3. Falls back to raw text if JSON parsing fails
  4. Cleans and trims extracted content
- **Output:** Clean, formatted section content text

**Step 6: Research Summary Generation (`_extract_research_summary`)**
- **Input:** Array of research findings from prompts
- **Process:**
  1. Iterates through research findings
  2. Formats each finding with question and key information
  3. Limits to top 3 pieces of information per finding
  4. Creates structured summary for LLM context
- **Output:** Formatted research summary string

**Step 7: Chapter Consolidation (`_generate_chapter_files`)**
- **Input:** Organized chapter data with sections
- **Process:**
  1. Groups sections by chapter title
  2. Creates chapter metadata (section count, word count, timestamp)
  3. Combines section content into full chapter text
  4. Saves both JSON (with metadata) and Markdown (content only) formats
  5. Generates chapter-level statistics
- **Output:** Chapter JSON and Markdown files

**Step 8: Report Summary Generation (`_generate_report_summary`)**
- **Input:** Complete processing results and chapter data
- **Process:**
  1. Calculates overall statistics (total chapters, sections, word count)
  2. Creates detailed chapter breakdown with section information
  3. Records output file locations and processing metadata
  4. Generates comprehensive processing summary
- **Output:** `report_summary.json` with complete generation statistics

#### Output Structure:

**File Outputs:**
- `sections/{section_name}.json` - Individual section files with content and metadata
- `chapters/{chapter_name}.json` - Consolidated chapter files with all sections
- `chapters/{chapter_name}.md` - Markdown chapter files for easy reading
- `report_summary.json` - Complete generation statistics and metadata

**Section JSON Structure:**
```json
{
  "section_metadata": {
    "chapter_title": "string",
    "section_title": "string",
    "description": "string",
    "word_count": "number",
    "generated_at": "ISO timestamp"
  },
  "content": "Generated section content text",
  "prompt_used": "Original LLM prompt",
  "processing_info": {
    "llm_model": "model identifier",
    "generation_time": "timestamp",
    "success": "boolean"
  }
}
```

**Chapter JSON Structure:**
```json
{
  "chapter_title": "string",
  "sections": [
    {
      "section_title": "string",
      "section_file": "path to section JSON",
      "content": "section content",
      "word_count": "number"
    }
  ],
  "total_sections": "number",
  "total_word_count": "number",
  "full_content": "Combined chapter content with headers",
  "generated_at": "ISO timestamp"
}
```

#### Key Configuration:
```python
# In writer_agent.py
class WriterAgent:
    def __init__(self, output_dir: str = "output/writer"):
        # Modify output directory structure
        self.output_dir = Path(output_dir)
        self.sections_dir = self.output_dir / "sections"
        self.chapters_dir = self.output_dir / "chapters"
        
    def process_section_prompts(self, prompts_dir: str) -> bool:
        # Main processing pipeline - modify workflow here
        
    def _create_llm_prompt(self, metadata: Dict, instructions: Dict, research: List[Dict]) -> str:
        # LLM prompt formatting - modify prompt structure here
        
    def _extract_content_from_response(self, raw_content: str) -> str:
        # Content extraction logic - modify parsing rules here
```

#### How to Modify:
- **Change output structure:** Modify `__init__` method directory setup and file naming
- **Customize LLM prompts:** Edit `_create_llm_prompt()` method template and instructions
- **Modify content processing:** Update `_extract_content_from_response()` parsing logic
- **Change chapter organization:** Edit `_generate_chapter_files()` consolidation rules
- **Add new metadata:** Extend section and chapter data structures
- **Customize research integration:** Modify `_extract_research_summary()` formatting
- **Change LLM settings:** Update LLM client configuration and model parameters

#### Error Handling:
- Validates existence of prompts directory and summary file
- Handles malformed JSON prompt files gracefully
- Manages LLM API failures with retry logic
- Provides detailed logging for each processing step
- Creates output directories automatically if they don't exist
- Tracks failed sections for debugging and reprocessing

#### Performance Considerations:
- Processes sections sequentially to manage LLM API rate limits
- Implements content caching to avoid regenerating existing sections
- Uses efficient file I/O with proper encoding handling
- Tracks processing statistics for performance monitoring
- Supports resume functionality for interrupted processing

---

### 5. PDF Generator Agent

**File:** `pdf_generator.py`
**Main Script:** `main_pdf_generator.py`

#### What it does:
- Combines all written chapters into a single PDF report
- Adds professional formatting, title page, and table of contents
- Creates a publication-ready document

#### Input:
- Completed chapters from Writer Agent
- Report structure for proper ordering
- Formatting and styling preferences

#### Process:
1. Reads all chapter files and report structure
2. Orders chapters according to the original plan
3. Converts Markdown content to PDF format
4. Adds title page, table of contents, and professional styling
5. Generates final PDF report

#### Output:
- `report_YYYYMMDD_HHMMSS.pdf`: Final PDF report
- Professional document ready for distribution

#### Key Configuration:
```python
# In pdf_generator.py
class PDFGenerator:
    def __init__(self, output_dir: str = "output/pdf"):
        # Modify output directory
        
    def generate_pdf_report(self, writer_dir: str, structure_path: str):
        # Modify PDF generation process
```

#### How to Modify:
- **Change PDF styling:** Edit `_create_reportlab_styles()` method
- **Modify page layout:** Update `_generate_reportlab_pdf()` method
- **Add new content types:** Extend `_html_to_flowables()` method
- **Customize title page:** Modify `_create_title_page_reportlab()` method

## Running the System

### Individual Agents

Run each agent separately:

```bash
# 1. Create plan
python main_planner.py --topic "Your Research Topic" --output-dir output/planner

# 2. Conduct research
python main_researcher.py --plan output/planner/plan.json --data-sources path/to/data --output-dir output/researcher

# 3. Generate section prompts
python main_section_report.py --research-dir output/researcher --plan output/planner/plan.json --output-dir output/section_prompts

# 4. Write report
python main_writer.py --prompts-dir output/section_prompts --research-dir output/researcher --output-dir output/writer

# 5. Generate PDF
python main_pdf_generator.py --writer-output output/writer --structure report_structure.json --output-dir output/pdf
```

### Complete Pipeline

Run the entire pipeline:

```bash
python main_pipeline.py --topic "Your Research Topic" --data-sources path/to/data --output-dir output
```

## Configuration Files

### `report_structure.json`
Defines the overall report structure, chapters, and sections. Modify this file to change the report organization.

### `llm_client.py`
Contains LLM configuration and API settings. Modify this to change the AI model or API endpoints.

## Customization Guide

### Adding New Section Types

1. **Planner Agent:** Add new section definitions in `_generate_sections()`
2. **Researcher Agent:** Add research methods for new section types
3. **Section Report Agent:** Create prompt templates for new sections
4. **Writer Agent:** Add content generation logic for new sections
5. **PDF Generator:** Add formatting rules for new content types

### Changing Output Formats

1. **Writer Agent:** Modify `_format_content()` to change output format
2. **PDF Generator:** Update `_html_to_flowables()` for new formatting

### Integrating New Data Sources

1. **Researcher Agent:** Add new data loading methods
2. Update `_load_data_sources()` to handle new source types

### Modifying AI Prompts

All AI prompts are embedded in the respective agent files. Search for prompt strings and modify them according to your needs.

## Troubleshooting

### Common Issues

1. **Missing Dependencies:** Install required packages from `requirements.txt`
2. **File Path Issues:** Ensure all paths use absolute paths or correct relative paths
3. **API Errors:** Check LLM API configuration in `llm_client.py`
4. **PDF Generation Errors:** Verify ReportLab installation and file permissions

### Debug Mode

Add `--log-level DEBUG` to any main script for detailed logging:

```bash
python main_planner.py --topic "Test" --log-level DEBUG
```

## File Structure

```
report_generator/
├── planner_agent.py          # Plan creation logic
├── researcher_agent.py       # Research functionality
├── section_report_agent.py   # Prompt generation
├── writer_agent.py          # Content writing
├── pdf_generator.py         # PDF creation
├── llm_client.py           # AI model interface
├── main_planner.py         # Planner CLI
├── main_researcher.py      # Researcher CLI
├── main_section_report.py  # Section Report CLI
├── main_writer.py          # Writer CLI
├── main_pdf_generator.py   # PDF Generator CLI
├── main_pipeline.py        # Complete pipeline
├── report_structure.json   # Report structure template
└── README.md              # This file
```

## Advanced Configuration

### Environment Variables

Set these environment variables for configuration:

```bash
export OPENAI_API_KEY="your-api-key"
export REPORT_OUTPUT_DIR="/path/to/output"
export LOG_LEVEL="INFO"
```

### Custom Styling

Modify PDF styles in `pdf_generator.py`:

```python
def _create_reportlab_styles(self):
    # Add your custom styles here
    styles['CustomStyle'] = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.blue
    )
```

This system provides a flexible, modular approach to automated report generation with full customization capabilities at each stage of the pipeline.