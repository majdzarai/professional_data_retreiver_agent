#!/usr/bin/env python3
"""
Relationship Extraction Module for Graph RAG Pipeline

This module provides AI-powered relationship extraction capabilities specifically designed
for AML/KYC compliance and financial regulatory document analysis. It identifies semantic
relationships between entities in unstructured text and converts them into structured
triples (source, relation, target) for knowledge graph construction.

Key Features:
- Semantic relationship extraction using Groq's fast language models
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
- GROQ_API_KEY environment variable
- Internet connection for API access
- Python packages: groq, python-dotenv

Author: Graph RAG Pipeline
Version: 1.0
Compatibility: Python 3.7+
"""

import os
import json
import re
import logging
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=API_KEY)

def extract_relationships(text_chunk: str) -> List[Dict[str, str]]:
    """
    Extract semantic relationships from text using Groq's high-performance language models.
    
    This function analyzes a text chunk and identifies meaningful relationships between
    entities, converting them into structured triples suitable for knowledge graph
    construction. It's specifically optimized for AML/KYC compliance documents and
    financial regulatory content.
    
    Processing Pipeline:
    1. Constructs specialized prompt for relationship extraction
    2. Sends request to Groq's Llama 3.1 8B Instant model
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
        - Requires valid GROQ_API_KEY environment variable
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
        logger.info(f"Sending text chunk to Groq API for relationship extraction (length: {len(text_chunk)} chars)")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Using Llama 3.1 8B Instant (current model)
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=1024  # Increased to handle longer responses
        )
        content = response.choices[0].message.content
        logger.info(f"Received response from Groq API (length: {len(content)} chars)")
        logger.info(f"Response content: {repr(content[:1000])}...")

        # Parse JSON safely
        try:
            data = json.loads(content)
            logger.info("Successfully parsed JSON directly from response")
        except Exception as direct_parse_error:
            logger.info(f"Direct JSON parsing failed: {direct_parse_error}")
            data = []
            
            # Try to extract JSON from markdown code blocks first
            # Look for ```json or ``` followed by JSON content
            json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', content, re.IGNORECASE)
            if json_match:
                try:
                    json_content = json_match.group(1).strip()
                    logger.info(f"Found JSON in markdown block (length: {len(json_content)}): {json_content[:200]}...")
                    data = json.loads(json_content)
                    logger.info(f"Successfully parsed JSON from markdown block: {len(data)} items")
                except Exception as e:
                    logger.warning(f"Failed to parse JSON from markdown block: {e}")
                    logger.warning(f"Markdown content was: {json_content[:500]}...")
                    
                    # Try to find just the JSON array part within the markdown block
                    array_match = re.search(r'(\[\s*\{[\s\S]*?\}\s*\])', json_content, re.S)
                    if array_match:
                        try:
                            array_content = array_match.group(1)
                            logger.info(f"Found JSON array within markdown: {array_content[:200]}...")
                            data = json.loads(array_content)
                            logger.info(f"Successfully parsed JSON array from markdown: {len(data)} items")
                        except Exception as array_e:
                            logger.warning(f"Failed to parse JSON array from markdown: {array_e}")
            
            if not data:
                # Fallback to finding any JSON array in the content
                match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", content, re.S)
                if match:
                    try:
                        json_content = match.group(0)
                        logger.info(f"Found JSON array with regex: {json_content[:200]}...")
                        data = json.loads(json_content)
                        logger.info(f"Successfully parsed JSON array: {len(data)} items")
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON array: {e}")
                        logger.warning(f"Array content was: {json_content[:500]}...")
            
            if not data:
                logger.warning(f"No valid JSON array found in response: {content[:500]}...")

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
