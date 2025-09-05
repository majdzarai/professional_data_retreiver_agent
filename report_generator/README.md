# Report Generator

A multi-agent system for generating comprehensive reports using RAG (Retrieval-Augmented Generation) technology.

## Overview

The Report Generator is a system that uses multiple AI agents to generate comprehensive reports based on a given report structure. The system leverages the existing RAG pipeline to retrieve relevant information from a document database and uses that information to generate well-structured, evidence-based reports.

## Architecture

The system consists of four main agents:

1. **Planner Agent**: Takes a report structure as input and generates a plan of questions and RAG queries for each section.
2. **Researcher Agent**: Executes RAG queries and collects evidence for each question.
3. **Writer Agent**: Generates draft content for each section based on the collected evidence.
4. **Editor Agent**: Compiles and edits the final report based on the section drafts.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main_report_generator.py --structure examples/sample_report_structure.json --output-dir output/reports
```

### Command-line Arguments

- `--structure`, `-s`: Path to the JSON file containing the report structure (required)
- `--output-dir`, `-o`: Directory to save the generated report and intermediate files (default: `output/reports`)
- `--verbose`, `-v`: Enable verbose logging
- `--model`: LLM model to use for the agents (default: `gpt-4`)
- `--rag-path`: Path to the RAG pipeline directory (default: `../rag`)

## Report Structure Format

The report structure is defined in a JSON file with the following format:

```json
{
  "title": "Report Title",
  "sections": [
    {
      "id": "section_id",
      "title": "Section Title",
      "description": "Section Description"
    },
    ...
  ]
}
```

## Output

The system generates the following outputs:

- **Plan**: JSON file containing the plan for each section, including questions and RAG queries.
- **Evidence**: JSON file containing the evidence collected for each question.
- **Section Drafts**: Markdown files containing the draft content for each section.
- **Final Report**: Markdown file containing the compiled report.

## Integration with RAG Pipeline

The Report Generator integrates with the existing RAG pipeline to retrieve relevant information from a document database. The Researcher Agent uses the RAG pipeline to execute queries and collect evidence for each question.

## Customization

The system can be customized by modifying the prompt templates for each agent. The prompt templates are defined in the `ReportGeneratorConfig` class in `utils/config.py`.

## License

[MIT License](LICENSE)