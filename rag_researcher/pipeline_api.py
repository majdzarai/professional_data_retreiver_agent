# rag_researcher/pipeline_api.py
import sys
import os
from pathlib import Path

# Add current directory and modules directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "modules"))

# Import from the current directory
from main_rag_pipeline import RAGPipeline, RAGPipelineConfig

# Create a single shared pipeline (loads existing vector store if available)
# This ensures we use the existing vector store from rag_researcher/output
output_dir = current_dir / "output"
config = RAGPipelineConfig(output_dir=str(output_dir))
pipeline = RAGPipeline(config=config)

def search_context(query: str, top_k: int = 5):
    """
    Run a retrieval query against the RAG pipeline.
    Returns top_k chunks with metadata.
    """
    results = pipeline.retrieve(query=query, top_k=top_k)
    return results
