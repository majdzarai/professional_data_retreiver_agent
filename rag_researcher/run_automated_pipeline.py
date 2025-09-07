#!/usr/bin/env python3
"""
Automated RAG Pipeline Runner

This script demonstrates how to use the automated RAG pipeline with folder monitoring.
It will continuously monitor an input folder, process new documents, and maintain
a persistent vector database.

Usage:
    python run_automated_pipeline.py
    
Or with custom parameters:
    python run_automated_pipeline.py --input custom_input --output custom_output --query "your query here"
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_rag_pipeline import AutomatedRAGPipeline

def main():
    parser = argparse.ArgumentParser(description='Run the Automated RAG Pipeline')
    parser.add_argument('--input', '-i', default='input', 
                       help='Input folder path (default: input)')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output folder path (default: output)')
    parser.add_argument('--query', '-q', default='company global information overview business operations',
                       help='Query to search for (default: company global information overview business operations)')
    parser.add_argument('--chunk-size', '-c', type=int, default=700,
                       help='Chunk size for text processing (default: 500)')
    parser.add_argument('--embedding-model', '-m', default='nomic-embed-text:latest',
                       help='Embedding model to use (default: nomic-embed-text:latest)')
    parser.add_argument('--embedding-provider', '-p', default='nomic',
                       help='Embedding provider to use (default: nomic)')
    parser.add_argument('--monitor-interval', '-t', type=int, default=30,
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--run-once', action='store_true',
                       help='Run pipeline once instead of continuous monitoring')
    
    args = parser.parse_args()
    
    # Create the pipeline
    pipeline = AutomatedRAGPipeline(
        input_dir=args.input,
        output_dir=args.output,
        vector_db_path=os.path.join(args.output, "vector_store.pkl"),
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        monitor_interval=args.monitor_interval
    )
    
    print(f"\n=== Automated RAG Pipeline ===")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Default query: {args.query}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Embedding provider: {args.embedding_provider}")
    print(f"Embedding model: {args.embedding_model}")
    
    if args.run_once:
        print(f"\nRunning pipeline once...")
        pipeline.run_once()
        print("\nPipeline completed. Check the output folder for results.")
    else:
        print(f"\nStarting continuous monitoring (interval: {args.monitor_interval}s)")
        print("Press Ctrl+C to stop the pipeline\n")
        
        # Show current status
        status = pipeline.status()
        print(f"Current status:")
        print(f"  - Vector database: {status['vector_db_exists']} ({status['document_count']} documents)")
        print(f"  - Input files: {status['input_files_count']}")
        print(f"  - Last processed: {status['last_processed']}")
        print()
        
        # Start continuous monitoring
        pipeline.run_continuous()

if __name__ == "__main__":
    main()