#!/usr/bin/env python3
"""
=============================================================================
                    GRAPH RAG RESEARCH PIPELINE
                   Advanced KYC/AML Compliance Analysis
=============================================================================

This is the main research pipeline that implements a complete Graph RAG 
(Retrieval-Augmented Generation) system for KYC/AML compliance analysis.
The pipeline processes financial documents and generates comprehensive 
intelligence reports using local AI models.

🔍 PIPELINE OVERVIEW:
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT: Financial Documents (PDF/TXT) → GRAPH RAG → OUTPUT: Intelligence │
└─────────────────────────────────────────────────────────────────────────┘

📊 WORKFLOW STAGES:
1. 📄 Text Extraction     - Extract clean text from documents
2. 🔪 Text Chunking       - Split text into manageable segments
3. 🏷️  Entity Extraction   - Identify key entities (companies, people, etc.)
4. 🔗 Relationship Mapping - Extract semantic relationships between entities
5. 🕸️  Graph Construction  - Build knowledge graph from entities/relationships
6. 👥 Community Detection - Identify clusters of related entities
7. 📋 Intelligence Reports - Generate professional KYC/AML summaries

🛡️  COMPLIANCE FEATURES:
- AML (Anti-Money Laundering) entity classification
- KYC (Know Your Customer) relationship mapping
- Risk assessment through community analysis
- Regulatory reporting with professional summaries

🔧 TECHNICAL STACK:
- Local AI: Llama 3.1 via Ollama (no external APIs)
- Graph Processing: NetworkX for graph operations
- NLP: Advanced entity and relationship extraction
- Output: JSON data + Human-readable reports

Author: AI Research Assistant
Version: 2.0
License: MIT
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os                    # Operating system interface for file operations
import sys                   # System-specific parameters and functions
import json                  # JSON encoder and decoder for data serialization
import logging              # Logging facility for Python applications
from pathlib import Path    # Object-oriented filesystem paths

# Add the graph_rag module to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'graph_rag'))

# Import specialized Graph RAG pipeline modules
from graph_rag.pdf_extractor import extract_text_from_pdf           # PDF text extraction
from graph_rag.chunking import chunk_text, save_chunks_to_file       # Text segmentation
from graph_rag.entity_extraction import extract_entities             # AI-powered entity recognition
from graph_rag.relationship_extraction import extract_relationships  # AI-powered relationship mapping
from graph_rag.graph_builder import build_graph                      # Knowledge graph construction
from graph_rag.community_detection import (                          # Community detection algorithms
    detect_communities_greedy, 
    save_communities, 
    build_community_index
)
from graph_rag.community_summary import make_community_summaries      # AI-powered report generation

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure comprehensive logging for pipeline monitoring
logging.basicConfig(
    level=logging.INFO,                                    # Set logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Detailed log format
)
logger = logging.getLogger(__name__)                       # Create logger instance

# =============================================================================
# SYSTEM HEALTH CHECKS
# =============================================================================

def check_ollama_server():
    """
    Verify that the Ollama AI server is running and has required models.
    
    This function performs a health check on the local AI infrastructure:
    - Connects to Ollama server on localhost:11434
    - Verifies Llama 3.1 models are available
    - Provides troubleshooting guidance if issues found
    
    Returns:
        bool: True if server is ready, False if setup required
    """
    import requests  # HTTP library for API calls
    
    try:
        # Attempt connection to Ollama API endpoint
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:  # Server responded successfully
            # Parse available models from API response
            models = response.json().get('models', [])
            # Filter for Llama 3.1 models specifically
            llama_models = [m for m in models if 'llama3.1' in m.get('name', '')]
            
            if llama_models:  # Required models found
                logger.info(f"✓ Ollama server is running with Llama 3.1 models: {[m['name'] for m in llama_models]}")
                return True
            else:  # Server running but missing models
                logger.error("✗ Ollama server is running but no Llama 3.1 models found")
                logger.info("Please install Llama 3.1: ollama pull llama3.1")
                return False
        else:  # Server responded with error
            logger.error(f"✗ Ollama server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:  # Connection failed
        logger.error(f"✗ Cannot connect to Ollama server: {e}")
        logger.info("Please start Ollama server: ollama serve")
        return False

# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_complete_pipeline(input_file: str, output_dir: str = "output_data"):
    """
    Execute the complete Graph RAG pipeline for KYC/AML analysis.
    
    This is the main orchestrator function that coordinates all pipeline stages:
    - Validates prerequisites and creates output structure
    - Processes input documents through all transformation stages
    - Generates comprehensive intelligence reports
    - Provides detailed progress monitoring and error handling
    
    Args:
        input_file (str): Path to input document (PDF or TXT format)
        output_dir (str): Directory for all pipeline outputs (default: "output_data")
        
    Returns:
        bool: True if pipeline completed successfully, False if errors occurred
    """
    # Display pipeline startup banner
    logger.info("=" * 60)
    logger.info("STARTING COMPLETE GRAPH RAG PIPELINE")
    logger.info("=" * 60)
    
    # =========================================================================
    # PREREQUISITE VALIDATION
    # =========================================================================
    
    # Verify AI infrastructure is ready
    if not check_ollama_server():
        logger.error("Pipeline cannot proceed without Ollama server")
        return False
    
    # =========================================================================
    # OUTPUT DIRECTORY STRUCTURE CREATION
    # =========================================================================
    
    # Create organized directory structure for all pipeline outputs
    os.makedirs(f"{output_dir}/cleaned_data", exist_ok=True)    # Processed text files
    os.makedirs(f"{output_dir}/chunks", exist_ok=True)          # Text segments
    os.makedirs(f"{output_dir}/entities", exist_ok=True)        # Extracted entities
    os.makedirs(f"{output_dir}/relationships", exist_ok=True)   # Entity relationships
    os.makedirs(f"{output_dir}/graph", exist_ok=True)           # Knowledge graphs
    os.makedirs(f"{output_dir}/communities", exist_ok=True)     # Community clusters
    os.makedirs(f"{output_dir}/summaries", exist_ok=True)       # Intelligence reports
    
    # Extract base filename for consistent naming across outputs
    input_path = Path(input_file)
    base_name = input_path.stem  # Filename without extension
    
    try:
        # =====================================================================
        # STAGE 1: DOCUMENT TEXT EXTRACTION
        # =====================================================================
        
        logger.info("\n📄 Stage 1: Text Extraction")
        
        if input_path.suffix.lower() == '.pdf':  # Handle PDF documents
            # Extract text content from PDF using specialized parser
            text_content = extract_text_from_pdf(input_file)
            # Save cleaned text for reference
            cleaned_file = f"{output_dir}/cleaned_data/{base_name}_cleaned.txt"
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            logger.info(f"✓ Extracted text from PDF: {len(text_content)} characters")
        else:  # Handle plain text files
            # Load text content directly
            with open(input_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            cleaned_file = input_file  # Use original file
            logger.info(f"✓ Loaded text file: {len(text_content)} characters")
        
        # =====================================================================
        # STAGE 2: INTELLIGENT TEXT CHUNKING
        # =====================================================================
        
        logger.info("\n🔪 Stage 2: Text Chunking")
        
        # Define output file for text chunks
        chunks_file = f"{output_dir}/chunks/{base_name}_chunks.txt"
        
        # Create semantically meaningful text segments
        chunks_data = chunk_text(text_content, source_filename=base_name)
        
        # Save chunks with metadata for reference
        save_chunks_to_file(chunks_data, chunks_file)
        
        # Extract text content for AI processing
        chunks = [chunk['text'] for chunk in chunks_data]
        logger.info(f"✓ Created {len(chunks)} text chunks")
        
        # =====================================================================
        # STAGE 3: AI-POWERED ENTITY EXTRACTION
        # =====================================================================
        
        logger.info("\n🏷️  Stage 3: Entity Extraction (Local Llama 3.1)")
        
        # Define output file for extracted entities
        entities_file = f"{output_dir}/entities/{base_name}_entities.json"
        
        # Process limited chunks for faster testing (can be expanded for production)
        test_chunks = chunks[:4]  # Process first chunk only for demo
        all_entities = []  # Accumulator for all discovered entities
        
        # Process each text chunk through AI entity extraction
        for i, chunk in enumerate(test_chunks):
            logger.info(f"Processing chunk {i+1}/{len(test_chunks)} (limited to 1 for testing)")
            # Use local Llama 3.1 to identify entities (companies, people, locations, etc.)
            chunk_entities = extract_entities(chunk)
            all_entities.extend(chunk_entities)  # Accumulate results
        
        # Save all discovered entities in structured JSON format
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(all_entities, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Extracted {len(all_entities)} entities using local Llama 3.1")
        
        # =====================================================================
        # STAGE 4: AI-POWERED RELATIONSHIP EXTRACTION
        # =====================================================================
        
        logger.info("\n🔗 Stage 4: Relationship Extraction (Local Llama 3.1)")
        
        # Define output file for entity relationships
        relationships_file = f"{output_dir}/relationships/{base_name}_relationships.json"
        
        # Process same chunks to extract semantic relationships
        all_relationships = []  # Accumulator for all discovered relationships
        
        # Process each text chunk through AI relationship extraction
        for i, chunk in enumerate(test_chunks):
            logger.info(f"Processing chunk {i+1}/{len(test_chunks)} for relationships (limited to 1 for testing)")
            # Use local Llama 3.1 to identify relationships (owns, partners_with, etc.)
            chunk_relationships = extract_relationships(chunk)
            all_relationships.extend(chunk_relationships)  # Accumulate results
        
        # Save all discovered relationships in structured JSON format
        with open(relationships_file, 'w', encoding='utf-8') as f:
            json.dump(all_relationships, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Extracted {len(all_relationships)} relationships using local Llama 3.1")
        
        # =====================================================================
        # STAGE 5: KNOWLEDGE GRAPH CONSTRUCTION
        # =====================================================================
        
        logger.info("\n🕸️  Stage 5: Graph Building")
        
        # Define output file for knowledge graph
        graph_file = f"{output_dir}/graph/{base_name}_graph.json"
        
        # Construct knowledge graph from entities and relationships
        build_graph(entities_file, relationships_file, graph_file)
        logger.info(f"✓ Built knowledge graph")
        
        # =====================================================================
        # STAGE 6: COMMUNITY DETECTION AND CLUSTERING
        # =====================================================================
        
        logger.info("\n👥 Stage 6: Community Detection")
        
        # Apply advanced clustering algorithms to identify entity communities
        communities = detect_communities_greedy(graph_file)
        
        # Define output file for community structure
        communities_file = f"{output_dir}/communities/{base_name}_communities.json"
        
        # Build mapping from nodes to their community assignments
        node_to_comm = build_community_index(communities)
        
        # Save community structure with metadata
        save_communities(communities, node_to_comm, communities_file)
        logger.info(f"✓ Detected {len(communities)} communities")
        
        # =====================================================================
        # STAGE 7: AI-POWERED INTELLIGENCE REPORT GENERATION
        # =====================================================================
        
        logger.info("\n📋 Stage 7: Community Summarization (Local Llama 3.1)")
        
        # Define output files for intelligence reports
        summaries_file = f"{output_dir}/summaries/{base_name}_summaries.json"      # Structured data
        summaries_txt = f"{output_dir}/summaries/{base_name}_summaries_report.txt"  # Human-readable report
        
        # Generate comprehensive KYC/AML intelligence reports
        make_community_summaries(
            communities_file=communities_file,  # Input: Community structure
            graph_file=graph_file,              # Input: Knowledge graph
            output_json=summaries_file,         # Output: Structured summaries
            output_txt=summaries_txt            # Output: Professional report
        )
        logger.info(f"✓ Generated community summaries using local Llama 3.1")
        
        # =====================================================================
        # PIPELINE COMPLETION AND RESULTS SUMMARY
        # =====================================================================
        
        # Display comprehensive completion summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        logger.info("=" * 60)
        logger.info(f"📁 All outputs saved to: {output_dir}/")
        logger.info(f"📊 Communities detected: {len(communities)}")
        logger.info(f"📄 Final summaries: {summaries_file}")
        
        # Display detailed community statistics
        logger.info("\n📈 Community Statistics:")
        for i, community in enumerate(communities):
            logger.info(f"  Community {i+1}: {len(community)} entities")
        
        return True  # Pipeline completed successfully
        
    except Exception as e:  # Handle any pipeline errors
        logger.error(f"❌ Pipeline failed at stage: {e}")
        logger.exception("Full error details:")  # Log complete stack trace
        return False  # Pipeline failed

# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================

def main():
    """
    Main execution function for the Graph RAG research pipeline.
    
    This function serves as the primary entry point when the script is run directly.
    It handles:
    - Input file validation
    - Pipeline execution coordination
    - Success/failure reporting
    - User guidance for setup issues
    """
    # =========================================================================
    # INPUT FILE CONFIGURATION
    # =========================================================================
    
    # Default input file path (modify as needed for your documents)
    input_file = "input_data/annual_report_text (1).txt"
    
    # Validate that input file exists before starting pipeline
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please place your PDF or text file in the input_data/ directory")
        return  # Exit if no input file
    
    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================
    
    # Execute the complete Graph RAG pipeline
    success = run_complete_pipeline(input_file)
    
    # =========================================================================
    # RESULTS REPORTING
    # =========================================================================
    
    if success:  # Pipeline completed successfully
        logger.info("\n🚀 Graph RAG pipeline test completed successfully!")
        logger.info("All stages used local processing - no external APIs required.")
    else:  # Pipeline encountered errors
        logger.error("\n💥 Pipeline test failed. Check the logs above for details.")

# =============================================================================
# SCRIPT EXECUTION GUARD
# =============================================================================

if __name__ == "__main__":
    # Execute main function only when script is run directly (not imported)
    main()