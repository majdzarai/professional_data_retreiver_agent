import json
import os
import networkx as nx
from networkx.algorithms import community as nx_comm
from typing import List, Dict, Any, Optional
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_graph(graph_path: str) -> nx.Graph:
    """
    Load a NetworkX graph from JSON file and convert to undirected graph.
    
    Args:
        graph_path (str): Path to the graph JSON file
        
    Returns:
        nx.Graph: Undirected NetworkX graph suitable for community detection
        
    Raises:
        FileNotFoundError: If the graph file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        Exception: For other graph loading errors
    """
    try:
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
            
        logger.info(f"Loading graph from: {graph_path}")
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Load as directed graph first, then convert to undirected
        # Most community detection algorithms work better with undirected graphs
        G_dir = nx.node_link_graph(data)
        G = G_dir.to_undirected()
        
        logger.info(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except FileNotFoundError:
        logger.error(f"Graph file not found: {graph_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in graph file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise

def detect_communities_greedy(graph_path: Optional[str] = None) -> List[List[str]]:
    """
    Detect communities using rule-based clustering for KYC/AML analysis.
    Creates 3-4 meaningful business clusters instead of micro-communities.
    
    Clusters:
    1. Global Operations: Companies, countries, products, hospitals, partners
    2. Financial Data: Revenue, margins, growth, employees, R&D, market metrics
    3. Compliance/KYC: Certifications, regulators, sanctions, ethics, transparency
    4. Other: Miscellaneous entities that don't fit above categories
    
    Args:
        graph_path (str, optional): Path to graph JSON file. 
                                  Defaults to test directory if None.
    
    Returns:
        List[List[str]]: List of 3-4 communities, where each community is a list of node IDs
        
    Raises:
        Exception: If community detection fails
    """
    if graph_path is None:
        # Default to test directory where our actual graph file is located
        graph_path = "test/knowledge_graph_2025-09-03_17-55-19.json"
        
    logger.info(f"Starting rule-based KYC/AML community detection on {graph_path}")
    
    try:
        G = _load_graph(graph_path)
        
        # Check if graph has any nodes
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, returning empty community list")
            return []
        
        # Initialize clusters
        global_cluster = []      # Companies, geography, products, operations
        financial_cluster = []   # Revenue, costs, growth, financial metrics
        compliance_cluster = []  # Regulations, certifications, ethics, compliance
        other_cluster = []       # Everything else
        
        # Define classification keywords and types
        global_keywords = {
            'company', 'corporation', 'business', 'organization', 'enterprise',
            'hospital', 'healthcare', 'medical', 'clinic', 'patient',
            'country', 'worldwide', 'global', 'international', 'region',
            'product', 'technology', 'solution', 'equipment', 'accelerator',
            'partner', 'client', 'customer', 'stakeholder', 'supplier'
        }
        
        financial_keywords = {
            'revenue', 'income', 'profit', 'margin', 'earnings', 'ebitda', 'rebit',
            'growth', 'percent', '%', 'million', 'eur', 'usd', 'dollar',
            'employee', 'staff', 'workforce', 'headcount',
            'r&d', 'research', 'development', 'investment',
            'market', 'share', 'sales', 'cost', 'expense'
        }
        
        compliance_keywords = {
            'certified', 'certification', 'iso', 'standard', 'quality',
            'b corp', 'bcorp', 'benefit corporation',
            'regulator', 'regulatory', 'compliance', 'audit',
            'sanction', 'embargo', 'restriction', 'penalty',
            'ethics', 'ethical', 'transparency', 'governance',
            'risk', 'control', 'policy', 'procedure'
        }
        
        # Global operation entity types
        global_types = {'COMPANY', 'COUNTRY', 'CITY', 'CLIENT', 'PRODUCT', 'MARKET'}
        
        # Financial entity types
        financial_types = {'AMOUNT', 'PERCENTAGE', 'METRIC'}
        
        # Compliance entity types
        compliance_types = {'REGULATOR', 'CERTIFICATION', 'STANDARD'}
        
        # Classify each node
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_type = node_data.get('type', 'UNKNOWN')
            node_text = str(node_id).lower()
            
            # Check entity type first (most reliable)
            if node_type in global_types:
                global_cluster.append(node_id)
                logger.debug(f"Node '{node_id}' → Global (type: {node_type})")
            elif node_type in financial_types:
                financial_cluster.append(node_id)
                logger.debug(f"Node '{node_id}' → Financial (type: {node_type})")
            elif node_type in compliance_types:
                compliance_cluster.append(node_id)
                logger.debug(f"Node '{node_id}' → Compliance (type: {node_type})")
            else:
                # Check keywords in node text
                if any(keyword in node_text for keyword in compliance_keywords):
                    compliance_cluster.append(node_id)
                    logger.debug(f"Node '{node_id}' → Compliance (keyword match)")
                elif any(keyword in node_text for keyword in financial_keywords):
                    financial_cluster.append(node_id)
                    logger.debug(f"Node '{node_id}' → Financial (keyword match)")
                elif any(keyword in node_text for keyword in global_keywords):
                    global_cluster.append(node_id)
                    logger.debug(f"Node '{node_id}' → Global (keyword match)")
                else:
                    other_cluster.append(node_id)
                    logger.debug(f"Node '{node_id}' → Other (no match)")
        
        # Build final communities list (only include non-empty clusters)
        communities = []
        cluster_names = []
        
        if global_cluster:
            communities.append(global_cluster)
            cluster_names.append("Global Operations")
        
        if financial_cluster:
            communities.append(financial_cluster)
            cluster_names.append("Financial Data")
        
        if compliance_cluster:
            communities.append(compliance_cluster)
            cluster_names.append("Compliance/KYC")
        
        if other_cluster:
            communities.append(other_cluster)
            cluster_names.append("Other")
        
        # Log results
        logger.info(f"✓ Rule-based clustering completed: {len(communities)} meaningful clusters")
        for i, (cluster, name) in enumerate(zip(communities, cluster_names)):
            logger.info(f"  Cluster {i}: {name} ({len(cluster)} nodes)")
        
        total_nodes = sum(len(comm) for comm in communities)
        logger.info(f"Total nodes classified: {total_nodes}/{G.number_of_nodes()}")
        
        return communities
        
    except Exception as e:
        logger.error(f"Error in community detection: {e}")
        raise

def detect_communities_connected_components(graph_path: Optional[str] = None) -> List[List[str]]:
    """
    Detect communities using connected components (simplest method).
    
    Each connected component becomes a separate community.
    This is useful for graphs with clear disconnected clusters.
    
    Args:
        graph_path (str, optional): Path to graph JSON file.
                                  Defaults to test directory if None.
    
    Returns:
        List[List[str]]: List of communities based on connected components
    """
    if graph_path is None:
        graph_path = "test/knowledge_graph_2025-09-03_17-55-19.json"
        
    try:
        G = _load_graph(graph_path)
        
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, returning empty community list")
            return []
            
        logger.info("Finding connected components...")
        comps = nx.connected_components(G)
        communities = [list(c) for c in comps]
        
        logger.info(f"Found {len(communities)} connected components")
        return communities
        
    except Exception as e:
        logger.error(f"Error in connected components detection: {e}")
        raise

def build_community_index(communities: List[List[str]]) -> Dict[str, int]:
    """
    Build a mapping from node ID to community ID.
    
    This creates a reverse index that allows quick lookup of which
    community a specific node belongs to.
    
    Args:
        communities (List[List[str]]): List of communities, each containing node IDs
        
    Returns:
        Dict[str, int]: Mapping from node_id to community_id
        
    Example:
        communities = [['A', 'B'], ['C', 'D']]
        returns: {'A': 0, 'B': 0, 'C': 1, 'D': 1}
    """
    try:
        idx = {}
        for cid, nodes in enumerate(communities):
            for node in nodes:
                if node in idx:
                    logger.warning(f"Node {node} appears in multiple communities")
                idx[node] = cid
                
        logger.info(f"Built community index for {len(idx)} nodes across {len(communities)} communities")
        return idx
        
    except Exception as e:
        logger.error(f"Error building community index: {e}")
        raise

def save_communities(
    communities: List[List[str]],
    node_to_comm: Dict[str, int],
    out_json: Optional[str] = None
) -> str:
    """
    Save community detection results to JSON file with KYC/AML cluster metadata.
    
    Args:
        communities (List[List[str]]): List of communities with node IDs
        node_to_comm (Dict[str, int]): Mapping from node ID to community ID
        out_json (str, optional): Output file path. Defaults to test directory.
        
    Returns:
        str: Path to the saved JSON file
        
    Raises:
        Exception: If file saving fails
    """
    if out_json is None:
        out_json = "test/communities.json"
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        
        # Calculate community statistics
        total_nodes = sum(len(comm) for comm in communities)
        avg_size = total_nodes / len(communities) if communities else 0
        
        # Define cluster names and descriptions for KYC/AML analysis
        cluster_metadata = [
            {
                "name": "Global Operations",
                "description": "Companies, countries, products, hospitals, partners, and operational entities",
                "kyc_relevance": "Entity identification, business relationships, geographic exposure"
            },
            {
                "name": "Financial Data", 
                "description": "Revenue, margins, growth metrics, employees, R&D, and financial indicators",
                "kyc_relevance": "Financial health assessment, transaction patterns, risk indicators"
            },
            {
                "name": "Compliance/KYC",
                "description": "Certifications, regulators, sanctions, ethics, transparency, and compliance matters",
                "kyc_relevance": "Regulatory compliance, sanctions screening, risk assessment"
            },
            {
                "name": "Other",
                "description": "Miscellaneous entities that don't fit primary business categories",
                "kyc_relevance": "Additional context and supporting information"
            }
        ]
        
        # Prepare data structure for JSON export with enhanced metadata
        payload = {
            "metadata": {
                "num_communities": len(communities),
                "total_nodes": total_nodes,
                "average_community_size": round(avg_size, 2),
                "largest_community_size": max(len(comm) for comm in communities) if communities else 0,
                "smallest_community_size": min(len(comm) for comm in communities) if communities else 0,
                "algorithm": "rule_based_kyc_aml",
                "version": "2.0",
                "purpose": "KYC/AML business intelligence clustering"
            },
            "communities": [
                {
                    "community_id": i, 
                    "nodes": comm,
                    "size": len(comm),
                    "cluster_info": cluster_metadata[i] if i < len(cluster_metadata) else {
                        "name": f"Cluster_{i}",
                        "description": "Additional cluster",
                        "kyc_relevance": "Supplementary information"
                    }
                } for i, comm in enumerate(communities)
            ],
            "node_to_community": node_to_comm
        }
        
        # Save to JSON file with proper formatting
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            
        logger.info(f"KYC/AML clusters saved to: {out_json}")
        logger.info(f"Summary: {len(communities)} communities, {total_nodes} total nodes")
        
        # Log cluster summary
        for i, comm in enumerate(communities):
            cluster_name = cluster_metadata[i]["name"] if i < len(cluster_metadata) else f"Cluster_{i}"
            logger.info(f"  {cluster_name}: {len(comm)} entities")
        
        return out_json
        
    except Exception as e:
        logger.error(f"Error saving communities: {e}")
        raise
