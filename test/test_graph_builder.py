#!/usr/bin/env python3
"""
Test Knowledge Graph Builder

This script builds a knowledge graph from the extracted entities and relationships,
combining them into nodes and edges for visualization and analysis.
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_rag.graph_builder import build_graph

def main():
    print("\n" + "="*70)
    print("🗄️  KNOWLEDGE GRAPH DATABASE BUILDER")
    print("="*70)
    print("📊 Building and displaying Knowledge Graph database...")
    print("-"*70)
    
    # Find the latest entity and relationship files
    test_dir = "test"
    entity_files = [f for f in os.listdir(test_dir) if (f.startswith("extracted_entities") or f.startswith("entities_output")) and f.endswith(".json")]
    relation_files = [f for f in os.listdir(test_dir) if (f.startswith("extracted_relations") or f.startswith("relations_output")) and f.endswith(".json")]
    
    if not entity_files:
        print("❌ No entity files found in test directory")
        return
    
    if not relation_files:
        print("❌ No relationship files found in test directory")
        return
    
    # Use the most recent files
    latest_entities = sorted(entity_files)[-1]
    latest_relations = sorted(relation_files)[-1]
    
    entities_path = os.path.join(test_dir, latest_entities)
    relations_path = os.path.join(test_dir, latest_relations)
    
    print(f"📄 Using entity file: {latest_entities}")
    print(f"🔗 Using relations file: {latest_relations}")
    
    # Load and preview the data
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    with open(relations_path, 'r', encoding='utf-8') as f:
        relations = json.load(f)
    
    print(f"\n📊 Data Summary:")
    print(f"   • Entities: {len(entities)}")
    print(f"   • Relationships: {len(relations)}")
    
    # Show entity type distribution
    entity_types = {}
    for entity in entities:
        etype = entity.get('type', 'UNKNOWN')
        entity_types[etype] = entity_types.get(etype, 0) + 1
    
    print(f"\n🏷️  Entity Types:")
    for etype, count in sorted(entity_types.items()):
        print(f"   • {etype}: {count}")
    
    # Show relationship type distribution
    relation_types = {}
    for relation in relations:
        rtype = relation.get('relation', 'unknown')
        relation_types[rtype] = relation_types.get(rtype, 0) + 1
    
    print(f"\n🔗 Relationship Types (top 10):")
    for rtype, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {rtype}: {count}")
    
    # Build the knowledge graph
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    graph_output = f"test/knowledge_graph_{timestamp}.json"
    
    print(f"\n🔨 Building knowledge graph...")
    
    try:
        result = build_graph(
            entities_file=entities_path,
            relations_file=relations_path,
            output_file=graph_output
        )
        print(f"✅ {result}")
        
        # Load and analyze the built graph
        with open(graph_output, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        nodes = graph_data.get('nodes', [])
        links = graph_data.get('links', [])
        
        print(f"\n📈 Graph Statistics:")
        print(f"   • Total nodes: {len(nodes)}")
        print(f"   • Total edges: {len(links)}")
        
        # Show most connected nodes
        node_connections = {}
        for link in links:
            source = link.get('source')
            target = link.get('target')
            node_connections[source] = node_connections.get(source, 0) + 1
            node_connections[target] = node_connections.get(target, 0) + 1
        
        print(f"\n🌟 Most Connected Entities (top 5):")
        for node, connections in sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {node}: {connections} connections")
        
        # Create a simple text-based graph visualization
        print(f"\n🕸️  Sample Graph Connections:")
        for i, link in enumerate(links[:10]):
            source = link.get('source')
            target = link.get('target')
            relation = link.get('relation', 'connected_to')
            print(f"   {i+1}. {source} --[{relation}]--> {target}")
        
        if len(links) > 10:
            print(f"   ... and {len(links) - 10} more connections")
        
        print(f"\n💾 Knowledge Graph Database saved to: {graph_output}")
        print(f"\n" + "="*70)
        print(f"🗄️  KNOWLEDGE GRAPH DATABASE SUMMARY")
        print(f"="*70)
        print(f"📋 Database Name: Knowledge Graph")
        print(f"📊 Database Type: Graph Database")
        print(f"🔢 Total Records: {len(nodes)} nodes, {len(links)} relationships")
        print(f"📁 Storage Location: {graph_output}")
        print(f"\n🎯 Knowledge Graph Database Features:")
        print(f"   • 🔍 Entity-Relationship Network Analysis")
        print(f"   • 📈 Graph-based Data Queries")
        print(f"   • 🌐 Network Visualization Support")
        print(f"   • 🧠 Knowledge Discovery & Insights")
        print(f"   • 🔗 Relationship Mapping & Analysis")
        print(f"\n✅ Knowledge Graph Database successfully created and ready for use!")
        
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        return

if __name__ == "__main__":
    main()
