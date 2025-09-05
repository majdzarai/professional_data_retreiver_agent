#!/usr/bin/env python
# Mock RAG pipeline script for testing

import sys
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Mock RAG pipeline")
    parser.add_argument("query", help="The query to execute")
    parser.add_argument("--reranking-enabled", action="store_true", help="Enable reranking")
    parser.add_argument("--hybrid-search-enabled", action="store_true", help="Enable hybrid search")
    parser.add_argument("--hybrid-search-weight", type=float, default=0.5, help="Hybrid search weight")
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    print(f"Executing query: {args.query}")
    print(f"Reranking enabled: {args.reranking_enabled}")
    print(f"Hybrid search enabled: {args.hybrid_search_enabled}")
    print(f"Hybrid search weight: {args.hybrid_search_weight}")

    # Return mock results
    results = {
        "query": args.query,
        "chunks": [
            {
                "text": f"Mock evidence chunk 1 for query: {args.query}",
                "source": "Mock Document A",
                "similarity_score": 0.85,
                "metadata": {"page": 1, "section": "Introduction"}
            },
            {
                "text": f"Mock evidence chunk 2 for query: {args.query}",
                "source": "Mock Document B",
                "similarity_score": 0.78,
                "metadata": {"page": 5, "section": "Analysis"}
            }
        ]
    }

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
