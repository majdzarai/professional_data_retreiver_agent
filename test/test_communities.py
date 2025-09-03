#!/usr/bin/env python3
"""
Community Detection and Summarization Test Script

This script demonstrates the complete community detection and analysis pipeline:
1. Load knowledge graph from test directory
2. Detect communities using greedy modularity algorithm
3. Generate comprehensive summaries using local LLM
4. Save results in both JSON and readable text formats

Author: Graph RAG Pipeline
Date: 2025
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from graph_rag.community_detection import (
    detect_communities_greedy,
    detect_communities_connected_components,
    build_community_index,
    save_communities
)
from graph_rag.community_summary import make_community_summaries

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_prerequisites() -> bool:
    """
    Check if all required files and services are available.
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    logger.info("Checking prerequisites...")
    
    # Check if knowledge graph file exists
    graph_file = "test/knowledge_graph_2025-09-03_17-55-19.json"
    if not os.path.exists(graph_file):
        logger.error(f"Knowledge graph file not found: {graph_file}")
        logger.error("Please run test_graph_builder.py first to create the knowledge graph")
        return False
    
    logger.info(f"✓ Knowledge graph file found: {graph_file}")
    
    # Check if test directory exists
    if not os.path.exists("test"):
        logger.error("Test directory not found")
        return False
    
    logger.info("✓ Test directory exists")
    
    # Check if Ollama service is running (optional - will fallback if not available)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama service is running - LLM summaries will be generated")
        else:
            logger.warning("⚠ Ollama service not responding - will use fallback summaries")
    except Exception as e:
        logger.warning(f"⚠ Cannot connect to Ollama service: {e} - will use fallback summaries")
    
    return True

def run_community_detection() -> tuple[list, dict, str]:
    """
    Run community detection on the knowledge graph.
    
    Returns:
        tuple: (communities, node_to_community_mapping, output_file_path)
        
    Raises:
        Exception: If community detection fails
    """
    logger.info("=" * 60)
    logger.info("STEP 1: COMMUNITY DETECTION")
    logger.info("=" * 60)
    
    try:
        # Method 1: Greedy modularity (recommended for most cases)
        logger.info("Running greedy modularity community detection...")
        communities = detect_communities_greedy()
        
        if not communities:
            logger.warning("No communities found with greedy method, trying connected components...")
            communities = detect_communities_connected_components()
        
        if not communities:
            raise Exception("No communities detected with any method")
        
        # Build reverse index for quick lookups
        node_to_comm = build_community_index(communities)
        
        # Save communities to JSON file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"test/communities_{timestamp}.json"
        saved_path = save_communities(communities, node_to_comm, output_file)
        
        # Log summary statistics
        total_nodes = sum(len(comm) for comm in communities)
        avg_size = total_nodes / len(communities)
        largest_comm = max(communities, key=len)
        smallest_comm = min(communities, key=len)
        
        logger.info(f"✓ Community detection completed successfully!")
        logger.info(f"  - Total communities: {len(communities)}")
        logger.info(f"  - Total nodes: {total_nodes}")
        logger.info(f"  - Average community size: {avg_size:.1f}")
        logger.info(f"  - Largest community: {len(largest_comm)} nodes")
        logger.info(f"  - Smallest community: {len(smallest_comm)} nodes")
        logger.info(f"  - Results saved to: {saved_path}")
        
        return communities, node_to_comm, saved_path
        
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise

def run_community_summarization(communities_file: str) -> tuple[str, str]:
    """
    Generate comprehensive summaries for detected communities.
    
    Args:
        communities_file (str): Path to the communities JSON file
        
    Returns:
        tuple: (json_output_path, text_output_path)
        
    Raises:
        Exception: If summarization fails
    """
    logger.info("=" * 60)
    logger.info("STEP 2: COMMUNITY SUMMARIZATION")
    logger.info("=" * 60)
    
    try:
        # Generate summaries using local LLM
        logger.info("Generating community summaries with local LLM...")
        
        json_path, txt_path = make_community_summaries(
            communities_file=communities_file
        )
        
        logger.info(f"✓ Community summarization completed successfully!")
        logger.info(f"  - JSON summaries: {json_path}")
        logger.info(f"  - Text report: {txt_path}")
        
        return json_path, txt_path
        
    except Exception as e:
        logger.error(f"Community summarization failed: {e}")
        raise

def display_results_summary(communities: list, json_path: str, txt_path: str):
    """
    Display a summary of the results to the user.
    
    Args:
        communities (list): List of detected communities
        json_path (str): Path to JSON summaries file
        txt_path (str): Path to text report file
    """
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Community size distribution
    sizes = [len(comm) for comm in communities]
    size_distribution = {}
    for size in sizes:
        size_range = f"{(size//5)*5}-{(size//5)*5+4}" if size < 20 else "20+"
        size_distribution[size_range] = size_distribution.get(size_range, 0) + 1
    
    logger.info("Community Size Distribution:")
    for size_range, count in sorted(size_distribution.items()):
        logger.info(f"  - {size_range} nodes: {count} communities")
    
    logger.info("\nGenerated Files:")
    logger.info(f"  - Communities JSON: {json_path}")
    logger.info(f"  - Analysis Report: {txt_path}")
    
    # Show file sizes
    try:
        json_size = os.path.getsize(json_path) / 1024  # KB
        txt_size = os.path.getsize(txt_path) / 1024   # KB
        logger.info(f"  - JSON file size: {json_size:.1f} KB")
        logger.info(f"  - Text file size: {txt_size:.1f} KB")
    except Exception:
        pass
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    print("\n🎉 Community Detection and Analysis Complete!")
    print(f"\n📊 Detected {len(communities)} communities from your knowledge graph")
    print(f"📄 Detailed analysis report: {txt_path}")
    print(f"💾 Machine-readable data: {json_path}")
    print("\n💡 Next steps:")
    print("   - Review the text report for insights")
    print("   - Import JSON data into visualization tools")
    print("   - Use communities for targeted graph queries")

def main():
    """
    Main function to run the complete community detection and analysis pipeline.
    """
    print("🚀 Starting Community Detection and Analysis Pipeline")
    print("=" * 70)
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites not met. Exiting.")
            sys.exit(1)
        
        # Step 1: Detect communities
        communities, node_mapping, communities_file = run_community_detection()
        
        # Step 2: Generate summaries
        json_path, txt_path = run_community_summarization(communities_file)
        
        # Step 3: Display results
        display_results_summary(communities, json_path, txt_path)
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Please check the logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
