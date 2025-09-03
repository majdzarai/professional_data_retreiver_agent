import os
import json
import requests
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
import logging
from datetime import datetime

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local LLM configuration (Ollama)
LOCAL_LLM_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "llama3.1:latest"  # Use the available model

def _summarize_with_local_llm(entities: List[str], sample_edges: List[str], community_id: int, cluster_name: str = "Unknown") -> str:
    """
    Generate a professional KYC/AML summary using local Ollama LLM.
    
    Args:
        entities (List[str]): List of entity names in the community
        sample_edges (List[str]): Sample relationships in the community
        community_id (int): ID of the community being summarized
        cluster_name (str): Name of the cluster (Global Operations, Financial Data, etc.)
        
    Returns:
        str: Generated KYC/AML summary or fallback summary if LLM fails
    """
    try:
        # Prepare entities and relationships for the prompt
        entities_text = ", ".join(entities[:15])  # Limit to first 15 entities
        if len(entities) > 15:
            entities_text += f" (and {len(entities) - 15} more)"
        
        relationships_text = "\n".join([
            f"- {edge}"
            for edge in sample_edges[:8]  # Limit to first 8 relationships
        ])
        
        # Create KYC/AML focused prompt based on cluster type
        if "Global" in cluster_name:
            focus_area = "entity identification, business relationships, and geographic exposure"
        elif "Financial" in cluster_name:
            focus_area = "financial health assessment, transaction patterns, and risk indicators"
        elif "Compliance" in cluster_name or "KYC" in cluster_name:
            focus_area = "regulatory compliance, sanctions screening, and risk assessment"
        else:
            focus_area = "general business intelligence and risk context"
        
        prompt = f"""You are a KYC/AML analyst reviewing the '{cluster_name}' cluster from a business knowledge graph.

Cluster Focus: {focus_area}

Entities ({len(entities)} total): {entities_text}

Key Relationships:
{relationships_text}

Provide a professional KYC/AML analysis in this exact format:

**THEME:** [Main business theme/area]

**KEY ENTITIES:** [Most important entities and their business roles]

**RELATIONSHIPS:** [Critical business relationships observed]

**RISK ASSESSMENT:** [Potential red flags, compliance considerations, or risk indicators]

Keep response under 250 words. Focus on KYC/AML relevance, compliance risks, and business intelligence."""
        
        # Prepare request for local Ollama LLM
        payload = {
            "model": LOCAL_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Lower temperature for more consistent professional output
                "num_predict": 400,  # Limit response length
                "top_p": 0.8
            }
        }
        
        logger.info(f"Generating KYC/AML summary for {cluster_name} cluster with {len(entities)} entities")
        
        # Make request to local LLM
        response = requests.post(LOCAL_LLM_URL, json=payload, timeout=45)  # Longer timeout for more detailed analysis
        response.raise_for_status()
        
        result = response.json()
        summary = result.get('response', '').strip()
        
        if summary:
            logger.info(f"✓ Generated KYC/AML summary for {cluster_name} cluster ({len(summary)} chars)")
            return summary
        else:
            logger.warning(f"Empty response from LLM for {cluster_name} cluster")
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"Cannot connect to Ollama service: {e}")
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
    
    # Professional fallback summary when LLM is unavailable
    risk_indicators = []
    entities_lower = entities_text.lower() if 'entities_text' in locals() else ' '.join(entities[:5]).lower()
    if "sanction" in entities_lower or "embargo" in entities_lower:
        risk_indicators.append("Potential sanctions exposure")
    if "regulator" in entities_lower or "compliance" in entities_lower:
        risk_indicators.append("Regulatory oversight required")
    if "revenue" in entities_lower and "million" in entities_lower:
        risk_indicators.append("Significant financial exposure")
    
    risk_text = "; ".join(risk_indicators) if risk_indicators else "Standard due diligence recommended"
    
    return f"""**THEME:** {cluster_name} - Business intelligence cluster (LLM service unavailable)

**KEY ENTITIES:** {len(entities)} entities including {', '.join(entities[:4])}{'...' if len(entities) > 4 else ''}

**RELATIONSHIPS:** {len(sample_edges)} business relationships identified across {cluster_name.lower()} domain

**RISK ASSESSMENT:** {risk_text}. Manual review required for comprehensive KYC/AML assessment. Cluster contains {len(entities)} entities with {len(sample_edges)} documented relationships requiring further analysis."""

def make_community_summaries(
    communities_file: str = "test/communities.json",
    graph_file: str = "test/knowledge_graph_2025-09-03_17-55-19.json",
    output_json: str = "test/summaries.json",
    output_txt: str = "test/summaries_report.txt",
    llm_model: str = "llama3.1:8b"
) -> Tuple[str, str]:
    """
    Generate comprehensive summaries for all detected communities.
    
    This function loads the knowledge graph and community data, then generates
    detailed summaries for each community using a local LLM.
    
    Args:
        graph_json (str, optional): Path to graph JSON file
        communities_json (str, optional): Path to communities JSON file  
        out_json (str, optional): Output path for summaries JSON
        out_txt (str, optional): Output path for readable text report
        
    Returns:
        tuple[str, str]: Paths to the generated JSON and text files
        
    Raises:
        FileNotFoundError: If required input files don't exist
        Exception: If summary generation fails
    """
    # Use provided file paths (defaults already set in function signature)
    graph_json = graph_file
    communities_json = communities_file
    out_json = output_json
    out_txt = output_txt
        
    try:
        # Validate input files exist
        if not os.path.exists(graph_json):
            raise FileNotFoundError(f"Graph file not found: {graph_json}")
        if not os.path.exists(communities_json):
            raise FileNotFoundError(f"Communities file not found: {communities_json}")
            
        logger.info(f"Loading graph from: {graph_json}")
        with open(graph_json, "r", encoding="utf-8") as f:
            gdata = json.load(f)
            
        # Create node lookup dictionary
        nodes = {n["id"]: n for n in gdata["nodes"]}
        edges = gdata["links"]
        
        logger.info(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges")
        
        logger.info(f"Loading communities from: {communities_json}")
        with open(communities_json, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            
        communities = cdata["communities"]
        logger.info(f"Processing {len(communities)} communities")
        
        results = []
        total_processed = 0
        
        for c in communities:
            cid = c["community_id"]
            nids = c["nodes"]
            
            logger.info(f"Processing community {cid} with {len(nids)} nodes")
            
            # Calculate entity type distribution
            types = Counter(nodes[n].get("type", "UNKNOWN") for n in nids if n in nodes)
            
            # Find all edges that involve nodes in this community
            community_edges = []
            internal_edges = []  # Edges within the community
            external_edges = []  # Edges connecting to other communities
            
            for e in edges:
                source_in_comm = e["source"] in nids
                target_in_comm = e["target"] in nids
                
                if source_in_comm or target_in_comm:
                    edge_text = f"{e['source']} —{e['relation']}→ {e['target']}"
                    community_edges.append(edge_text)
                    
                    if source_in_comm and target_in_comm:
                        internal_edges.append(edge_text)
                    else:
                        external_edges.append(edge_text)
            
            # Get cluster name from community metadata if available
            cluster_name = "Unknown Cluster"
            if 'cluster_info' in c:
                cluster_name = c['cluster_info'].get('name', f"Cluster_{cid}")
            else:
                # Fallback: determine cluster name based on entity types and content
                if "COMPANY" in types or "COUNTRY" in types or "CLIENT" in types:
                    cluster_name = "Global Operations"
                elif "AMOUNT" in types or "PERCENTAGE" in types:
                    cluster_name = "Financial Data"
                elif "REGULATOR" in types or "CERTIFICATION" in types:
                    cluster_name = "Compliance/KYC"
                else:
                    cluster_name = "Other"
            
            # Generate summary using local LLM
            try:
                summary = _summarize_with_local_llm(nids, community_edges, cid, cluster_name)
            except Exception as e:
                logger.warning(f"Failed to generate LLM summary for community {cid}: {e}")
                summary = f"Community {cid} analysis failed. Manual review required."
            
            # Compile comprehensive community data
            community_result = {
                "community_id": cid,
                "size": len(nids),
                "type_counts": dict(types),
                "summary": summary,
                "sample_nodes": nids[:25],
                "statistics": {
                    "total_edges": len(community_edges),
                    "internal_edges": len(internal_edges),
                    "external_edges": len(external_edges),
                    "connectivity_ratio": round(len(internal_edges) / max(len(community_edges), 1), 3)
                },
                "top_entity_types": [f"{k}: {v}" for k, v in types.most_common(5)]
            }
            
            results.append(community_result)
            total_processed += 1
            
        logger.info(f"Generated summaries for {total_processed} communities")
        
        # Create output directories
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        os.makedirs(os.path.dirname(out_txt), exist_ok=True)
        
        # Save comprehensive JSON results
        summary_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_communities": len(results),
                "source_graph": graph_json,
                "source_communities": communities_json,
                "llm_model": LOCAL_LLM_MODEL,
                "report_type": "KYC_AML_Risk_Assessment",
                "analysis_focus": "Business Intelligence and Compliance Risk"
            },
            "risk_summary": {
                "total_entities": sum(r['size'] for r in results),
                "high_risk_clusters": len([r for r in results if r['size'] > (sum(r['size'] for r in results) / len(results)) * 1.5]),
                "average_cluster_size": round(sum(r['size'] for r in results) / len(results), 1) if results else 0
            },
            "summaries": results
        }
        
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved JSON summaries to: {out_json}")
        
        # Generate readable text report
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("KYC/AML BUSINESS INTELLIGENCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Communities: {len(results)}\n")
            f.write(f"Source Graph: {graph_json}\n")
            f.write(f"LLM Model: {LOCAL_LLM_MODEL}\n")
            f.write(f"Report Type: KYC/AML Risk Assessment\n\n")
            
            # Summary statistics
            total_nodes = sum(r['size'] for r in results)
            avg_size = total_nodes / len(results) if results else 0
            largest_comm = max(results, key=lambda x: x['size']) if results else None
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Business Entities: {total_nodes}\n")
            f.write(f"Average Cluster Size: {avg_size:.1f}\n")
            if largest_comm:
                f.write(f"Largest Risk Cluster: {largest_comm['community_id']} ({largest_comm['size']} entities)\n")
            
            # Add risk summary
            high_risk_clusters = [r for r in results if r['size'] > avg_size * 1.5]
            f.write(f"High-Priority Clusters: {len(high_risk_clusters)} (requiring enhanced due diligence)\n")
            f.write("\n")
            
            # Individual cluster details with risk assessment
            for r in results:
                risk_level = "HIGH" if r['size'] > avg_size * 1.5 else "MEDIUM" if r['size'] > avg_size else "LOW"
                
                f.write(f"CLUSTER {r['community_id']} - RISK LEVEL: {risk_level}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Entity Count: {r['size']} entities\n")
                f.write(f"Entity Distribution: {r['type_counts']}\n")
                f.write(f"Network Analysis: {r['statistics']['internal_edges']} internal, {r['statistics']['external_edges']} external connections\n")
                f.write(f"Primary Types: {', '.join(r['top_entity_types'])}\n")
                f.write(f"Connectivity Ratio: {r['statistics']['connectivity_ratio']}\n\n")
                f.write("KYC/AML ASSESSMENT:\n")
                f.write(r["summary"] + "\n")
                f.write("\n" + "=" * 80 + "\n\n")
                
        logger.info(f"Saved text report to: {out_txt}")
        
        return out_json, out_txt
        
    except FileNotFoundError:
        logger.error(f"Required input files not found")
        raise
    except Exception as e:
        logger.error(f"Error generating community summaries: {e}")
        raise
