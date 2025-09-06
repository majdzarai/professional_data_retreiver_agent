#!/usr/bin/env python3
"""
Relationship Extraction Module for Graph RAG Pipeline

This module provides AI-powered relationship extraction capabilities specifically designed
for AML/KYC compliance and financial regulatory document analysis. It identifies semantic
relationships between entities in unstructured text and converts them into structured
triples (source, relation, target) for knowledge graph construction.

Key Features:
- Semantic relationship extraction using local Llama 3.1 models via Ollama
- Structured triple output format for graph databases
- Robust JSON parsing with fallback mechanisms
- Data normalization and cleaning
- Comprehensive error handling and logging
- Optimized for compliance and regulatory documents

Relationship Types Extracted:
- Business operations (operates, manages, owns)
- Financial relationships (fined, paid, invested)
- Regulatory actions (investigated, sanctioned, approved)
- Corporate structure (subsidiary_of, acquired, merged)
- Temporal relationships (reported_in, occurred_on)
- Quantitative relationships (employs, generated, lost)

Output Format:
    Each relationship is represented as a dictionary with three keys:
    {
        "source": "Entity that performs the action",
        "relation": "Type of relationship/action", 
        "target": "Entity that receives the action or object"
    }

Typical Usage:
    >>> from graph_rag.relationship_extraction import extract_relationships
    >>> text = "Wells Fargo was fined $3 million by the SEC in 2023."
    >>> relationships = extract_relationships(text)
    >>> print(relationships)
    [{'source': 'SEC', 'relation': 'fined', 'target': 'Wells Fargo'},
     {'source': 'Wells Fargo', 'relation': 'paid', 'target': '$3 million'}]

Requirements:
- Local Ollama server running with Llama 3.1 model
- Python packages: requests

Author: Graph RAG Pipeline
Version: 1.0
Compatibility: Python 3.7+
"""

import json
import re
import logging
import requests
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama server configuration for local Llama 3.1
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

def extract_relationships(text_chunk: str) -> List[Dict[str, str]]:
    """
    Extract semantic relationships from text using local Llama 3.1 model.
    
    This function analyzes a text chunk and identifies meaningful relationships between
    entities, converting them into structured triples suitable for knowledge graph
    construction. It's specifically optimized for AML/KYC compliance documents and
    financial regulatory content.
    
    Processing Pipeline:
    1. Constructs specialized prompt for relationship extraction
    2. Sends request to local Llama 3.1 via Ollama
    3. Parses JSON response with robust error handling
    4. Normalizes and cleans relationship data
    5. Validates triple structure and content
    
    Args:
        text_chunk (str): Text segment to analyze for relationships.
                         Optimal size: 200-800 tokens for best accuracy.
                         Should contain coherent content with clear entity interactions.
    
    Returns:
        List[Dict[str, str]]: List of relationship triples, where each dictionary contains:
            - "source" (str): Entity that performs the action or is the subject
            - "relation" (str): Normalized relationship type (lowercase, cleaned)
            - "target" (str): Entity that receives the action or is the object
            
    Example:
        >>> text = "IBA reported 42% growth and employed 1,986 people in 2024."
        >>> relationships = extract_relationships(text)
        >>> print(relationships)
        [{'source': 'IBA', 'relation': 'reported', 'target': '42% growth'},
         {'source': 'IBA', 'relation': 'employed', 'target': '1,986 people'}]
    
    Note:
        - Returns empty list if extraction fails (network, API, or parsing errors)
        - Relationships are normalized (lowercase relations, trimmed whitespace)
        - Invalid or incomplete triples are filtered out automatically
        - Uses temperature=0.1 for consistent, factual extraction
        - Requires local Ollama server running with Llama 3.1 model
    """
    prompt = f"""
    Extract ALL semantic relationships for comprehensive AML/KYC compliance and regulatory reporting from the following text.
    Return STRICT JSON array of objects with keys: "source", "relation", "target".

    Extract these relationship types with maximum coverage:

    **OWNERSHIP & CONTROL:**
    - OWNS, CONTROLS, MANAGES, OPERATES, GOVERNS
    - HOLDS_STAKE_IN, OWNS_SHARES_IN, HAS_INTEREST_IN
    - SUBSIDIARY_OF, PARENT_OF, AFFILIATED_WITH
    - ACQUIRED, MERGED_WITH, SOLD_TO, DIVESTED

    **EMPLOYMENT & GOVERNANCE:**
    - EMPLOYS, WORKS_FOR, REPORTS_TO, MANAGES
    - DIRECTOR_OF, EXECUTIVE_OF, BOARD_MEMBER_OF
    - APPOINTED_TO, RESIGNED_FROM, TERMINATED_FROM
    - FOUNDED, ESTABLISHED, CREATED

    **BUSINESS RELATIONSHIPS:**
    - PARTNERS_WITH, COLLABORATES_WITH, JOINT_VENTURE_WITH
    - SUPPLIES_TO, PROVIDES_TO, SERVICES, CONTRACTS_WITH
    - CLIENT_OF, CUSTOMER_OF, VENDOR_OF
    - COMPETES_WITH, RIVAL_OF

    **FINANCIAL RELATIONSHIPS:**
    - INVESTS_IN, FUNDS, FINANCES, LENDS_TO
    - RECEIVES_FUNDING_FROM, BORROWS_FROM
    - PAYS, RECEIVES_PAYMENT_FROM, TRANSACTS_WITH
    - FINED_BY, PENALIZED_BY, SANCTIONED_BY

    **REGULATORY & LEGAL:**
    - REGULATED_BY, SUPERVISED_BY, OVERSEEN_BY
    - LICENSED_BY, AUTHORIZED_BY, CERTIFIED_BY
    - COMPLIES_WITH, VIOLATES, BREACHES
    - INVESTIGATED_BY, AUDITED_BY, EXAMINED_BY

    **GEOGRAPHIC & OPERATIONAL:**
    - LOCATED_IN, BASED_IN, OPERATES_IN, PRESENT_IN
    - HEADQUARTERED_IN, INCORPORATED_IN, REGISTERED_IN
    - SERVES_MARKET, ACTIVE_IN, EXPANDS_TO

    **TEMPORAL & DEVELOPMENTAL:**
    - ESTABLISHED_IN, FOUNDED_IN, STARTED_IN
    - ANNOUNCED, PLANNED, LAUNCHED, COMPLETED
    - REPORTED, DISCLOSED, PUBLISHED, FILED

    **RISK & COMPLIANCE:**
    - EXPOSED_TO, SUBJECT_TO, AFFECTED_BY
    - MONITORS, ASSESSES, EVALUATES, REVIEWS
    - IMPLEMENTS, ADOPTS, FOLLOWS, MAINTAINS

    Text:
    {text_chunk}
    
    Extract every possible relationship - be comprehensive and thorough for regulatory analysis.
    """

    try:
        logger.info(f"Sending text chunk to local Llama 3.1 for relationship extraction (length: {len(text_chunk)} chars)")
        
        # Prepare the request payload for Ollama
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 1.0,
                "num_predict": 2048  # Increased from 1024 to handle longer responses
            }
        }
        
        # Make HTTP POST request to Ollama
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)  # 2 minute timeout
        response.raise_for_status()
        
        # Extract the actual text content from the Ollama response
        response_data = response.json()
        content = response_data.get("response", "")
        logger.info(f"Received response from local Llama 3.1 (length: {len(content)} chars)")
        logger.info(f"Response content (first 500 chars): {repr(content[:500])}...")

        # Parse JSON safely
        data = None
        try:
            data = json.loads(content)
            logger.info("Successfully parsed JSON directly from response")
        except Exception as direct_parse_error:
            logger.info(f"Direct JSON parsing failed: {direct_parse_error}")
            
            # Try to extract JSON from markdown code blocks first
            # Look for ```json or ``` followed by JSON content
            logger.info("Attempting to extract JSON from markdown blocks...")
            
            # Try multiple patterns for markdown extraction
            markdown_patterns = [
                r'```json\s*\n([\s\S]*?)\n```',  # ```json with newlines
                r'```json([\s\S]*?)```',         # ```json without newlines
                r'```\s*\n([\s\S]*?)\n```',     # ``` with newlines
                r'```([\s\S]*?)```'              # ``` without newlines
            ]
            
            for i, pattern in enumerate(markdown_patterns):
                json_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if json_match:
                    try:
                        json_content = json_match.group(1).strip()
                        logger.info(f"Pattern {i+1} matched! JSON content length: {len(json_content)}")
                        logger.info(f"JSON content preview: {json_content[:200]}...")
                        data = json.loads(json_content)
                        logger.info(f"Successfully parsed JSON from markdown block: {len(data)} items")
                        break
                    except Exception as e:
                        logger.warning(f"Pattern {i+1} matched but JSON parsing failed: {e}")
                        logger.warning(f"Content was: {json_content[:300]}...")
                        continue
                else:
                    logger.info(f"Pattern {i+1} did not match")
            
            if not data:
                # Try to find the first complete JSON array in the response
                # This handles cases where LLM returns JSON followed by explanatory text
                
                # Find the start of a JSON array
                start_match = re.search(r'\[', content)
                if start_match:
                    start_pos = start_match.start()
                    
                    # Find the matching closing bracket by counting brackets
                    bracket_count = 0
                    end_pos = start_pos
                    
                    for i, char in enumerate(content[start_pos:], start_pos):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
                    
                    if bracket_count == 0:  # Found complete array
                        try:
                            json_content = content[start_pos:end_pos]
                            logger.info(f"Found complete JSON array: {json_content[:200]}...")
                            data = json.loads(json_content)
                            logger.info(f"Successfully parsed complete JSON array: {len(data)} items")
                        except Exception as e:
                            logger.warning(f"Failed to parse complete JSON array: {e}")
                            
                            # Fallback to regex patterns
                            array_patterns = [
                                r'\[\s*\{[\s\S]*?"source"[\s\S]*?"relation"[\s\S]*?"target"[\s\S]*?\}[\s\S]*?\]',
                                r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]'
                            ]
                            
                            for pattern in array_patterns:
                                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                                if match:
                                    try:
                                        json_content = match.group(0)
                                        logger.info(f"Found JSON array with pattern: {json_content[:200]}...")
                                        data = json.loads(json_content)
                                        logger.info(f"Successfully parsed JSON array: {len(data)} items")
                                        break
                                    except Exception as e:
                                        logger.warning(f"Failed to parse JSON array with pattern: {e}")
                                        continue
            
            if not data:
                logger.warning(f"No valid JSON array found in response: {content[:500]}...")
                data = []  # Ensure data is a list for the normalization step

        # Normalize
        cleaned = []
        for rel in data:
            if all(k in rel for k in ["source", "relation", "target"]):
                cleaned.append({
                    "source": str(rel["source"]).strip(),
                    "relation": str(rel["relation"]).lower().strip(),
                    "target": str(rel["target"]).strip()
                })
        
        logger.info(f"Successfully extracted {len(cleaned)} relationships")
        return cleaned

    except Exception as e:
        logger.error(f"❌ Relationship extraction failed: {e}")
        return []
