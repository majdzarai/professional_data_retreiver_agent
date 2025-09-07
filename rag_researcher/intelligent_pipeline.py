#!/usr/bin/env python3
"""
Intelligent RAG Pipeline

Automatically detects scenario and handles:
- Document processing (when input files exist)
- Query retrieval (always returns top 5 results)
- Smart decision making based on current state
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from automated_rag_pipeline import AutomatedRAGPipeline

class IntelligentRAGPipeline:
    def __init__(self, input_dir="input", output_dir="output", vector_db_path=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.vector_db_path = vector_db_path or str(self.output_dir / "vector_store.pkl")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
   # Initialize the automated pipeline with Nomic embedding model
        self.pipeline = AutomatedRAGPipeline(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            vector_db_path=self.vector_db_path,
            embedding_provider="nomic",
            embedding_model="nomic-embed-text:latest",
            chunk_size=500,
            chunk_overlap=50
        )
    
    def detect_scenario(self):
        """
        Detect which scenario we're in:
        1. Has input files + No vector DB = Process documents, create DB
        2. Has input files + Has vector DB = Process new documents, update DB
        3. No input files + Has vector DB = Query only mode
        """
        has_input_files = self._has_input_files()
        has_vector_db = self._has_vector_db()
        
        if has_input_files and not has_vector_db:
            return "process_create_db"
        elif has_input_files and has_vector_db:
            return "process_update_db"
        elif not has_input_files and has_vector_db:
            return "query_only"
        else:
            return "no_data"
    
    def _has_input_files(self):
        """Check if input directory has any files"""
        if not self.input_dir.exists():
            return False
        
        # Check for common document types
        file_extensions = ['.txt', '.pdf', '.docx', '.md', '.json']
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                return True
        return False
    
    def _has_vector_db(self):
        """Check if vector database exists"""
        return Path(self.vector_db_path).exists()
    
    def run_intelligent_pipeline(self, query=None, top_k=5):
        """
        Main intelligent pipeline that:
        1. Detects current scenario
        2. Processes documents if needed
        3. Retrieves results if query provided
        """
        scenario = self.detect_scenario()
        self.logger.info(f"Detected scenario: {scenario}")
        
        if scenario == "no_data":
            self.logger.error("No input files and no vector database found!")
            print("❌ Error: No documents to process and no existing vector database.")
            print("Please add documents to the 'input' folder first.")
            return None
        
        # Handle processing scenarios
        if scenario in ["process_create_db", "process_update_db"]:
            self.logger.info("Processing documents...")
            print(f"📁 Found input files. Processing documents...")
            
            try:
                # Process documents
                self.pipeline.run_once()
                print("✅ Document processing completed successfully!")
                
                # Update scenario after processing
                scenario = "query_ready"
                
            except Exception as e:
                self.logger.error(f"Error during processing: {e}")
                print(f"❌ Error during processing: {e}")
                return None
        
        # Handle query if provided
        if query and scenario in ["query_only", "query_ready"]:
            self.logger.info(f"Retrieving results for query: '{query}'")
            print(f"🔍 Searching for: '{query}'")
            
            try:
                # Retrieve results
                results = self.pipeline.retrieve(query, top_k=top_k)
                
                if isinstance(results, list) and results:
                    print(f"\n✅ Found {len(results)} relevant results:\n")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        score = result.get('similarity_score', 0)
                        content = result.get('text', result.get('content', 'N/A'))
                        filename = result.get('filename', 'Unknown')
                        
                        print(f"📄 Result {i} (Score: {score:.4f})")
                        print(f"   File: {filename}")
                        print(f"   Content: {str(content)[:200]}...")
                        print()
                    
                    return results
                else:
                    print("❌ No relevant results found for your query.")
                    return []
                    
            except Exception as e:
                self.logger.error(f"Error during retrieval: {e}")
                print(f"❌ Error during retrieval: {e}")
                return None
        
        elif query:
            print("⚠️  Query provided but no vector database available after processing.")
            return None
        
        else:
            print("✅ Processing completed. Provide a query to search the documents.")
            return "processed"
    
    def get_status(self):
        """Get current pipeline status"""
        scenario = self.detect_scenario()
        input_files = list(self.input_dir.glob('*')) if self.input_dir.exists() else []
        input_count = len([f for f in input_files if f.is_file()])
        
        status = {
            "scenario": scenario,
            "input_files_count": input_count,
            "has_vector_db": self._has_vector_db(),
            "vector_db_path": self.vector_db_path,
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir)
        }
        
        return status

def main():
    parser = argparse.ArgumentParser(description="Intelligent RAG Pipeline")
    parser.add_argument("--query", "-q", type=str, help="Query to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--input-dir", type=str, default="input", help="Input directory path")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory path")
    parser.add_argument("--vector-db", type=str, help="Vector database path")
    parser.add_argument("--status", action="store_true", help="Show current pipeline status")
    
    args = parser.parse_args()
    
    # Initialize intelligent pipeline
    pipeline = IntelligentRAGPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vector_db_path=args.vector_db
    )
    
    if args.status:
        # Show status
        status = pipeline.get_status()
        print("\n📊 Pipeline Status:")
        print(f"   Scenario: {status['scenario']}")
        print(f"   Input files: {status['input_files_count']}")
        print(f"   Vector DB exists: {status['has_vector_db']}")
        print(f"   Vector DB path: {status['vector_db_path']}")
        print(f"   Input directory: {status['input_directory']}")
        print(f"   Output directory: {status['output_directory']}")
        return
    
    # Run intelligent pipeline
    print("🚀 Starting Intelligent RAG Pipeline...")
    print(f"   Input Directory: {args.input_dir}")
    print(f"   Output Directory: {args.output_dir}")
    if args.query:
        print(f"   Query: '{args.query}'")
        print(f"   Top-K Results: {args.top_k}")
    print()
    
    results = pipeline.run_intelligent_pipeline(
        query=args.query,
        top_k=args.top_k
    )
    
    if results == "processed":
        print("\n💡 Tip: Run again with --query 'your question' to search the processed documents.")
    elif results:
        print(f"\n🎯 Retrieved {len(results)} results successfully!")
        print("📁 Detailed results saved in the output/retriever folder.")

if __name__ == "__main__":
    main()