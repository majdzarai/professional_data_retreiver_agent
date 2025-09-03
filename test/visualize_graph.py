#!/usr/bin/env python3
"""
Knowledge Graph Visualization and Analysis

This script demonstrates how to analyze and visualize the knowledge graph
built from extracted entities and relationships.
"""

import json
import os
import sys
from collections import defaultdict, Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_latest_graph():
    """Load the most recent knowledge graph file."""
    test_dir = "test"
    graph_files = [f for f in os.listdir(test_dir) if f.startswith("knowledge_graph") and f.endswith(".json")]
    
    if not graph_files:
        print("❌ No knowledge graph files found")
        return None
    
    latest_graph = sorted(graph_files)[-1]
    graph_path = os.path.join(test_dir, latest_graph)
    
    print(f"📊 Loading graph: {latest_graph}")
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_graph_structure(graph_data):
    """Analyze the structure and properties of the knowledge graph."""
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    print(f"\n🔍 GRAPH STRUCTURE ANALYSIS")
    print(f"="*50)
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(links)}")
    
    # Node type analysis
    node_types = Counter(node.get('type', 'UNKNOWN') for node in nodes)
    print(f"\n📋 Node Types:")
    for ntype, count in node_types.most_common():
        print(f"   • {ntype}: {count} nodes")
    
    # Relationship type analysis
    relation_types = Counter(link.get('relation', 'unknown') for link in links)
    print(f"\n🔗 Relationship Types:")
    for rtype, count in relation_types.most_common():
        print(f"   • {rtype}: {count} edges")
    
    return nodes, links

def find_central_entities(nodes, links):
    """Find the most central/connected entities in the graph."""
    # Calculate degree centrality (number of connections)
    node_degrees = defaultdict(int)
    
    for link in links:
        source = link.get('source')
        target = link.get('target')
        node_degrees[source] += 1
        node_degrees[target] += 1
    
    print(f"\n🌟 CENTRAL ENTITIES (Top 10)")
    print(f"="*50)
    
    for i, (entity, degree) in enumerate(sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        # Find entity type
        entity_type = "UNKNOWN"
        for node in nodes:
            if node.get('id') == entity:
                entity_type = node.get('type', 'UNKNOWN')
                break
        
        print(f"{i:2d}. {entity} ({entity_type}) - {degree} connections")
    
    return node_degrees

def analyze_relationship_patterns(links):
    """Analyze patterns in relationships."""
    print(f"\n🕸️  RELATIONSHIP PATTERNS")
    print(f"="*50)
    
    # Group relationships by source entity
    source_relations = defaultdict(list)
    for link in links:
        source = link.get('source')
        relation = link.get('relation')
        target = link.get('target')
        source_relations[source].append((relation, target))
    
    # Show relationship patterns for top entities
    for source, relations in list(source_relations.items())[:3]:
        print(f"\n📍 {source}:")
        relation_counts = Counter(rel for rel, _ in relations)
        for relation, count in relation_counts.most_common(5):
            targets = [target for rel, target in relations if rel == relation]
            print(f"   • {relation} ({count}x): {', '.join(targets[:3])}{'...' if len(targets) > 3 else ''}")

def create_graph_summary(graph_data):
    """Create a comprehensive summary of the knowledge graph."""
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    # Basic statistics
    total_nodes = len(nodes)
    total_edges = len(links)
    
    # Calculate graph density
    max_possible_edges = total_nodes * (total_nodes - 1)  # for directed graph
    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    # Find isolated nodes (nodes with no connections)
    connected_nodes = set()
    for link in links:
        connected_nodes.add(link.get('source'))
        connected_nodes.add(link.get('target'))
    
    isolated_nodes = total_nodes - len(connected_nodes)
    
    print(f"\n📈 GRAPH SUMMARY")
    print(f"="*50)
    print(f"Nodes: {total_nodes}")
    print(f"Edges: {total_edges}")
    print(f"Graph Density: {density:.4f} ({density*100:.2f}%)")
    print(f"Connected Nodes: {len(connected_nodes)}")
    print(f"Isolated Nodes: {isolated_nodes}")
    print(f"Average Degree: {(total_edges * 2) / total_nodes:.2f}" if total_nodes > 0 else "Average Degree: 0")

def export_graph_for_visualization(graph_data):
    """Export graph in formats suitable for visualization tools."""
    print(f"\n💾 EXPORT OPTIONS")
    print(f"="*50)
    
    # Create a simple CSV export for nodes
    nodes_csv = "test/graph_nodes.csv"
    with open(nodes_csv, 'w', encoding='utf-8') as f:
        f.write("id,type,source\n")
        for node in graph_data.get('nodes', []):
            node_id = node.get('id', '').replace('"', '""')
            node_type = node.get('type', '')
            source = node.get('source', '')
            f.write(f'"{node_id}","{node_type}","{source}"\n')
    
    # Create a simple CSV export for edges
    edges_csv = "test/graph_edges.csv"
    with open(edges_csv, 'w', encoding='utf-8') as f:
        f.write("source,target,relation,source_doc\n")
        for link in graph_data.get('links', []):
            source = link.get('source', '').replace('"', '""')
            target = link.get('target', '').replace('"', '""')
            relation = link.get('relation', '')
            source_doc = link.get('source', '')
            f.write(f'"{source}","{target}","{relation}","{source_doc}"\n')
    
    print(f"✅ Exported nodes to: {nodes_csv}")
    print(f"✅ Exported edges to: {edges_csv}")
    print(f"\n🎯 You can now import these CSV files into:")
    print(f"   • Gephi (network visualization)")
    print(f"   • Cytoscape (biological networks)")
    print(f"   • NetworkX (Python analysis)")
    print(f"   • D3.js (web visualization)")
    print(f"   • Neo4j (graph database)")

def main():
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH VISUALIZATION & ANALYSIS")
    print("="*60)
    
    # Load the graph
    graph_data = load_latest_graph()
    if not graph_data:
        return
    
    # Perform analysis
    nodes, links = analyze_graph_structure(graph_data)
    node_degrees = find_central_entities(nodes, links)
    analyze_relationship_patterns(links)
    create_graph_summary(graph_data)
    export_graph_for_visualization(graph_data)
    
    print(f"\n✅ Analysis complete! Your knowledge graph is ready for:")
    print(f"   🔍 Further analysis and querying")
    print(f"   📊 Visualization in external tools")
    print(f"   🤖 AI-powered graph reasoning")
    print(f"   🔎 Knowledge discovery and insights")

if __name__ == "__main__":
    main()