#!/usr/bin/env python3
"""
Complete Entity & Relationship Extraction Pipeline Test with PDF Processing

This script:
1. Runs PDF extractor to process files from pdf_data/ and input_data/ folders
2. Cleans and saves processed text to output_data/cleaned_data/
3. Loads cleaned documents for chunking
4. Extracts both entities and relationships from each chunk
5. Saves the results to the test/ folder
"""

import os
import json
import sys
from datetime import datetime

# Add the parent directory to the path so we can import from graph_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag.chunking import load_documents, chunk_text, save_chunks_to_file
from graph_rag.entity_extraction import extract_entities
from graph_rag.relationship_extraction import extract_relationships
from graph_rag.pdf_extractor import process_all_files_in_folders

def test_entity_relationship_pipeline():
    print("=" * 70)
    print("COMPLETE ENTITY + RELATIONSHIP EXTRACTION PIPELINE TEST")
    print("=" * 70)

    # Define paths
    pdf_dir = "pdf_data"
    input_dir = "input_data"
    cleaned_dir = "output_data/cleaned_data"
    test_dir = "test"

    os.makedirs(test_dir, exist_ok=True)

    # 1. PDF extraction
    try:
        successful, failed = process_all_files_in_folders(
            pdf_folder=pdf_dir, txt_folder=input_dir, output_folder=cleaned_dir
        )
        print(f"✓ PDF extraction done, {successful} files processed, {failed} failed")
    except Exception as e:
        print(f"⚠️ PDF extractor error: {e}, continuing with existing cleaned files")

    # 2. Load cleaned documents
    documents = load_documents(cleaned_dir)
    if not documents:
        print("❌ No documents found")
        return

    print(f"✓ Loaded {len(documents)} documents")

    all_entities, all_relations = [], []
    total_chunks = 0

    # 3. Process each doc
    for filename, content in documents:
        print(f"\n📄 Document: {filename}")
        chunks = chunk_text(content)
        total_chunks += len(chunks)
        print(f"   → {len(chunks)} chunks created")
        
        # Save all chunks to file for inspection
        chunks_output_path = os.path.join(test_dir, "chunks_output.txt")
        save_chunks_to_file(chunks, chunks_output_path)
        print(f"   → All chunks saved to: {chunks_output_path}")

        chunks_to_process = min(5, len(chunks))  # limit for testing
        print(f"   → Processing first {chunks_to_process} chunks only")

        for i, chunk in enumerate(chunks[:chunks_to_process]):
            print(f"   🔎 Chunk {i+1}/{chunks_to_process}")

            # Entity extraction
            entities = extract_entities(chunk["text"])
            for ent in entities:
                ent.update({
                    "source_document": filename,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_metadata": chunk["metadata"]
                })
            all_entities.extend(entities)

            # Relationship extraction
            relations = extract_relationships(chunk["text"])
            for rel in relations:
                rel.update({
                    "source_document": filename,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_metadata": chunk["metadata"]
                })
            all_relations.extend(relations)

            print(f"      → {len(entities)} entities, {len(relations)} relations")

    # 4. Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    entities_path = os.path.join(test_dir, f"extracted_entities_{timestamp}.json")
    relations_path = os.path.join(test_dir, f"extracted_relations_{timestamp}.json")

    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(all_entities, f, indent=2, ensure_ascii=False)
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump(all_relations, f, indent=2, ensure_ascii=False)

    print("\n✅ Extraction complete")
    print(f"   Entities saved to: {entities_path}")
    print(f"   Relations saved to: {relations_path}")
    print(f"   Total chunks processed: {total_chunks}")
    print(f"   Total entities: {len(all_entities)}")
    print(f"   Total relations: {len(all_relations)}")

if __name__ == "__main__":
    test_entity_relationship_pipeline()
