#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.embedder import TextEmbedder
import numpy as np

def test_ollama_nomic():
    """Test Ollama-based Nomic embedding"""
    print("Testing Ollama Nomic embedding...")
    
    # Test with nomic-embed-text model via Ollama
    embedder = TextEmbedder(
        model_name="nomic-embed-text",
        model_provider="nomic",
        embedding_dim=768
    )
    
    # Test single embedding
    test_text = "This is a test sentence for embedding."
    print(f"\nTesting text: '{test_text}'")
    
    embedding = embedder.embed_text(test_text)
    if embedding is not None:
        print(f"✓ Embedding successful!")
        print(f"  Shape: {np.array(embedding).shape}")
        print(f"  Type: {type(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Embedding dimension: {len(embedding)}")
    else:
        print("✗ Embedding failed")
        return False
    
    # Test batch embedding
    test_texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    print(f"\nTesting batch embedding with {len(test_texts)} texts...")
    batch_embeddings = embedder._batch_embed(test_texts)
    
    if batch_embeddings is not None and len(batch_embeddings) > 0:
        print(f"✓ Batch embedding successful!")
        print(f"  Number of embeddings: {len(batch_embeddings)}")
        print(f"  Shape of first embedding: {np.array(batch_embeddings[0]).shape}")
        print(f"  First embedding first 3 values: {batch_embeddings[0][:3]}")
    else:
        print("✗ Batch embedding failed")
        return False
    
    return True

if __name__ == "__main__":
    success = test_ollama_nomic()
    if success:
        print("\n🎉 All tests passed! Ollama Nomic embedding is working correctly.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)