#!/usr/bin/env python3
"""
Knowledge Graph Builder Module for Graph RAG Pipeline

This module constructs knowledge graphs from extracted entities and relationships,
combining them into a unified NetworkX graph structure suitable for analysis,
visualization, and querying. It serves as the final assembly step in the
Graph RAG pipeline, converting structured data into a queryable knowledge base.

Key Features:
- Combines entities and relationships into directed graphs
- NetworkX integration for graph algorithms and analysis
- JSON export in node-link format for interoperability
- Metadata preservation for traceability
- Automatic node and edge creation with conflict resolution
- Support for multiple data sources and document tracking

Graph Structure:
- Nodes: Represent entities with type and source metadata
- Edges: Represent relationships with relation type and source metadata
- Directed: Relationships have clear source → target directionality
- Weighted: Can be extended to include relationship strength/confidence

Output Formats:
- NetworkX DiGraph object for programmatic analysis
- JSON node-link format for visualization tools (D3.js, Gephi, Cytoscape)
- Compatible with graph databases (Neo4j, ArangoDB)

Typical Usage:
    >>> from graph_rag.graph_builder import build_graph
    >>> result = build_graph(
    ...     entities_file="output_data/entities.json",
    ...     relations_file="output_data/relations.json",
    ...     output_file="output_data/knowledge_graph.json"
    ... )
    >>> print(result)
    ✅ Graph built: 52 nodes, 37 edges → output_data/knowledge_graph.json

Requirements:
- Python packages: networkx
- Input files: entities.json and relations.json from extraction pipeline

Author: Graph RAG Pipeline
Version: 1.0
Compatibility: Python 3.7+
"""

import json
import os
import networkx as nx

def build_graph(
    entities_file="output_data/entities.json",
    relations_file="output_data/relations.json",
    output_file="output_data/graph.json"
):
    """
    Construct a comprehensive knowledge graph from extracted entities and relationships.
    
    This function serves as the main graph assembly pipeline, combining structured
    entity and relationship data into a unified NetworkX directed graph. The resulting
    graph preserves all metadata and provides a foundation for advanced analytics,
    visualization, and knowledge discovery.
    
    Processing Pipeline:
    1. Initialize empty directed graph (NetworkX DiGraph)
    2. Load and process entity data, creating nodes with metadata
    3. Load and process relationship data, creating edges with metadata
    4. Handle missing nodes automatically (from relationships)
    5. Export graph in JSON node-link format for interoperability
    6. Generate processing statistics and confirmation
    
    Args:
        entities_file (str): Path to JSON file containing extracted entities.
                           Expected format: List of dicts with 'name', 'type', 'source_document'
                           Default: "output_data/entities.json"
        relations_file (str): Path to JSON file containing extracted relationships.
                            Expected format: List of dicts with 'source', 'target', 'relation', 'source_document'
                            Default: "output_data/relations.json"
        output_file (str): Path where the final graph JSON will be saved.
                         Uses NetworkX node-link format for maximum compatibility.
                         Default: "output_data/graph.json"
    
    Returns:
        str: Success message with graph statistics in format:
             "✅ Graph built: {nodes} nodes, {edges} edges → {output_file}"
    
    Graph Structure:
        Nodes (Entities):
        - id: Entity name (unique identifier)
        - type: Entity category (COMPANY, PERSON, BANK, etc.)
        - source: Source document for traceability
        
        Edges (Relationships):
        - source: Source entity name
        - target: Target entity name  
        - relation: Relationship type (normalized)
        - source: Source document for traceability
    
    Example:
        >>> result = build_graph(
        ...     entities_file="test/entities_2025-01-03.json",
        ...     relations_file="test/relations_2025-01-03.json",
        ...     output_file="test/knowledge_graph.json"
        ... )
        >>> print(result)
        ✅ Graph built: 52 nodes, 37 edges → test/knowledge_graph.json
    
    Note:
        - Creates output directory automatically if it doesn't exist
        - Handles missing input files gracefully (creates empty graph)
        - Automatically adds missing nodes referenced in relationships
        - Preserves all metadata for downstream analysis and debugging
        - Output JSON is compatible with D3.js, Gephi, Cytoscape, and Neo4j import tools
    """
    G = nx.DiGraph()

    # Load entities
    if os.path.exists(entities_file):
        with open(entities_file, "r", encoding="utf-8") as f:
            entities = json.load(f)
        for e in entities:
            G.add_node(e["name"], type=e["type"], source=e.get("source_document", ""))
    else:
        entities = []

    # Load relationships
    if os.path.exists(relations_file):
        with open(relations_file, "r", encoding="utf-8") as f:
            relations = json.load(f)
        for r in relations:
            src, tgt, rel_type = r["source"], r["target"], r["relation"]
            # Skip relationships with None/null values
            if src is None or tgt is None or src == "" or tgt == "":
                continue
            G.add_node(src)
            G.add_node(tgt)
            G.add_edge(src, tgt, relation=rel_type, source=r.get("source_document", ""))
    else:
        relations = []

    # Save graph
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = nx.node_link_data(G)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return f"✅ Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges → {output_file}"
