#!/usr/bin/env python3
"""
Entity Extraction Module for Graph RAG Pipeline

This module provides AI-powered named entity recognition (NER) capabilities specifically
designed for AML/KYC compliance and financial regulatory document analysis. It leverages
Groq's high-performance cloud-based language models to extract structured entities from
unstructured text with high accuracy and speed.

Key Features:
- AML/KYC-specific entity types (COMPANY, PERSON, BANK, REGULATOR, etc.)
- Groq cloud API integration for fast, reliable inference
- Robust error handling and response parsing
- Entity validation and deduplication
- Comprehensive logging for debugging and monitoring
- Type-safe implementation with full type hints

Supported Entity Types:
- COMPANY: Business entities, corporations, commercial organizations
- PERSON: Individual names (customers, executives, beneficial owners)
- BANK: Financial institutions, banks, credit unions
- COUNTRY: Countries, nations, jurisdictions
- REGULATOR: Regulatory bodies, government agencies (SEC, FINRA, FCA)
- DATE: Specific dates, years, time periods
- AMOUNT: Monetary amounts, transaction values, fines
- LAW: Laws, regulations, acts, compliance frameworks
- OTHER: Significant entities not covered by specific categories


Requirements:
- GROQ_API_KEY environment variable (obtain from https://console.groq.com/)
- Internet connection for API access
- Python packages: groq, python-dotenv

Author: Graph RAG Pipeline
Version: 1.0
Compatibility: Python 3.7+
"""

# Import necessary libraries for JSON handling, logging, type hints, environment variables, and Groq API
import json  # For parsing JSON responses from the AI model
import logging  # For debugging and tracking what the code is doing
import os  # For accessing environment variables like API keys
from typing import List, Dict, Any  # For type hints to make code more readable
from groq import Groq  # The Groq client for making API calls to Groq's AI models
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file (this reads GROQ_API_KEY)
load_dotenv()

# Configure logging so we can see what's happening during entity extraction
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO for detailed output
logger = logging.getLogger(__name__)  # Create a logger specific to this file

# Initialize the Groq client with API key from environment variables
# This client will be used to make requests to Groq's AI models
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")  # Get the API key from .env file
)

def extract_entities(text_chunk: str) -> List[Dict[str, str]]:
    """
    Extract named entities from a text chunk using Groq's AI models.
    
    This function is designed specifically for AML/KYC compliance investigations.
    It uses Groq's cloud-based AI models to identify and extract key entities that are
    commonly relevant in financial compliance and regulatory documents.
    
    Args:
        text_chunk (str): A chunk of text to analyze for named entities.
                         Should be a reasonably sized chunk (200-500 tokens) for best results.
    
    Returns:
        List[Dict[str, str]]: A list of extracted entities, where each entity is a dictionary with:
            - "name" (str): The actual entity text as found in the document
            - "type" (str): The entity category (COMPANY, PERSON, BANK, COUNTRY, REGULATOR, DATE, AMOUNT, LAW, OTHER)
    
   
    Note:
        - Requires valid GROQ_API_KEY in .env file
        - Handles API response parsing errors gracefully
        - Returns empty list if extraction fails
    """
    try:
        # Build the prompt that will be sent to the AI model
        # This prompt gives the AI specific instructions on what entities to find
        prompt = f"""You are an expert in AML/KYC compliance and financial due diligence. Extract ALL relevant named entities from the following text for comprehensive risk assessment and regulatory reporting.

Extract these entity types with maximum coverage:

**CORPORATE ENTITIES:**
- COMPANY: Corporations, LLCs, partnerships, subsidiaries, holding companies, joint ventures
- SUBSIDIARY: Child companies, divisions, business units, branches
- COMPETITOR: Rival companies, market competitors
- SUPPLIER: Vendors, contractors, service providers
- CLIENT: Customers, clients, business partners

**PEOPLE & ROLES:**
- PERSON: Full names of individuals (executives, directors, employees, beneficial owners)
- DIRECTOR: Board members, executive directors, non-executive directors
- EXECUTIVE: CEOs, CFOs, CTOs, presidents, vice presidents, managing directors
- EMPLOYEE: Staff members, consultants, advisors
- SHAREHOLDER: Equity holders, investors, stakeholders

**FINANCIAL & LEGAL:**
- BANK: Financial institutions, investment banks, credit unions, payment processors
- AMOUNT: Monetary values, revenues, profits, investments, fines, penalties
- CURRENCY: USD, EUR, GBP, and other currency codes
- REGULATOR: SEC, FINRA, FCA, CFTC, regulatory bodies, government agencies
- LAW: Acts, regulations, compliance frameworks, legal requirements
- LICENSE: Permits, certifications, authorizations, registrations

**GEOGRAPHIC & TEMPORAL:**
- COUNTRY: Nations, jurisdictions, territories
- CITY: Cities, towns, municipalities
- ADDRESS: Street addresses, postal codes, office locations
- DATE: Specific dates, years, quarters, reporting periods
- PERIOD: Time ranges, fiscal years, contract terms

**BUSINESS OPERATIONS:**
- PRODUCT: Services, products, offerings, solutions
- MARKET: Industries, sectors, market segments
- TECHNOLOGY: Software, platforms, systems, technologies
- PROJECT: Initiatives, programs, developments, acquisitions
- CONTRACT: Agreements, deals, partnerships, transactions
- RISK: Risk factors, compliance issues, regulatory concerns

Text to analyze:
{text_chunk}

Return ONLY a valid JSON array of objects. Each object must have exactly two fields: "name" and "type".
Extract every possible entity - be comprehensive and thorough.

JSON:"""
        
        # Log that we're about to make an API call (helpful for debugging)
        logger.info(f"Sending text chunk to Groq API for entity extraction (length: {len(text_chunk)} chars)")
        
        # Make the API call to Groq
        # We use the chat completions endpoint with specific parameters
        chat_completion = client.chat.completions.create(
            messages=[  # The conversation format - we send one user message with our prompt
                {
                    "role": "user",  # We are the user asking the AI to extract entities
                    "content": prompt  # Our carefully crafted prompt with the text to analyze
                }
            ],
            model="openai/gpt-oss-120b",  # The specific Groq model to use (as requested)
            temperature=0.1,  # Low temperature (0.1) for consistent, factual extraction (not creative)
            max_tokens=2048,  # Maximum tokens in the response (enough for entity lists)
            top_p=1,  # Use all possible tokens (standard setting)
            stream=False  # Get the complete response at once (not streaming)
        )
        
        # Extract the actual text content from the API response
        # The response has a specific structure, we need to get the message content
        response_text = chat_completion.choices[0].message.content
        
        # Log the response for debugging (show first 200 characters)
        logger.info(f"Received response from Groq API: {response_text[:200]}...")
        
        # Parse the JSON response using our helper function
        # This handles cases where the AI might return extra text around the JSON
        entities = _parse_llm_response(response_text)
        
        # Validate and clean the entities using our helper function
        # This ensures all entities have proper format and valid types
        validated_entities = _validate_entities(entities)
        
        # Log how many entities we successfully extracted
        logger.info(f"Successfully extracted {len(validated_entities)} entities")
        return validated_entities
        
    except Exception as e:
        # If anything goes wrong (API error, network issue, etc.), log it but don't crash
        logger.error(f"Error during entity extraction: {str(e)}")
        logger.error(f"Text chunk that caused error: {text_chunk[:100]}...")  # Show problematic text
        return []  # Return empty list so the pipeline can continue

def _parse_llm_response(response_text: str) -> List[Dict[str, str]]:
    """
    Parse the AI model response and extract JSON entities.
    
    The AI model might return extra text around the JSON, so this function
    attempts to find and parse the JSON array from the response.
    This is a helper function that handles various response formats gracefully.
    
    Args:
        response_text (str): Raw response from the AI model
        
    Returns:
        List[Dict[str, str]]: Parsed entities or empty list if parsing fails
    """
    try:
        # First, try to parse the response directly as JSON
        # This works if the AI returned clean JSON without extra text
        entities = json.loads(response_text.strip())  # Remove whitespace and parse
        if isinstance(entities, list):  # Make sure it's a list (array) as expected
            return entities
        else:
            # If it's not a list, log a warning and return empty list
            logger.warning("AI response is not a JSON array")
            return []
            
    except json.JSONDecodeError:
        # If direct parsing fails, the AI probably included extra text
        logger.info("Direct JSON parsing failed, attempting to extract JSON from response")
        
        # Import regex module for pattern matching
        import re
        
        # Look for JSON array patterns in the response using regex
        # This pattern matches: [ { ... } ] (JSON array with objects)
        json_pattern = r'\[\s*\{[^\]]+\]'
        matches = re.findall(json_pattern, response_text, re.DOTALL)  # Find all matches
        
        # Try to parse each potential JSON match
        for match in matches:
            try:
                entities = json.loads(match)  # Try to parse this match
                if isinstance(entities, list):  # If it's a valid list
                    logger.info("Successfully extracted JSON from AI response")
                    return entities
            except json.JSONDecodeError:
                continue  # If this match doesn't work, try the next one
        
        # If no complete JSON array found, try to extract individual entity objects
        # This pattern matches individual entity objects: {"name": "...", "type": "..."}
        entity_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"[^"]+"\s*\}'
        entity_matches = re.findall(entity_pattern, response_text)
        
        # If we found individual entities, try to parse them
        if entity_matches:
            entities = []  # Start with empty list
            for entity_str in entity_matches:  # For each entity string found
                try:
                    entity = json.loads(entity_str)  # Parse individual entity
                    entities.append(entity)  # Add to our list
                except json.JSONDecodeError:
                    continue  # Skip invalid entities
            
            # If we successfully parsed some entities, return them
            if entities:
                logger.info(f"Extracted {len(entities)} individual entities from response")
                return entities
        
        # If nothing worked, log a warning and return empty list
        logger.warning("Could not parse any valid JSON from AI response")
        return []

def _validate_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validate and clean the extracted entities.
    
    Ensures each entity has the required fields and valid entity types.
    Filters out invalid or malformed entities to ensure data quality.
    This is a helper function that cleans up the AI's output.
    
    Args:
        entities (List[Dict[str, str]]): Raw entities from AI model
        
    Returns:
        List[Dict[str, str]]: Validated and cleaned entities
    """
    # Define the valid entity types for AML/KYC compliance
    # Only these types are allowed - anything else gets changed to 'OTHER'
    valid_types = {
        # Corporate Entities
        'COMPANY', 'CORPORATION', 'LLC', 'PARTNERSHIP', 'TRUST', 'FOUNDATION',
        'SUBSIDIARY', 'HOLDING_COMPANY', 'JOINT_VENTURE', 'BRANCH', 'REPRESENTATIVE_OFFICE',
        
        # People & Roles
        'PERSON', 'DIRECTOR', 'CEO', 'CFO', 'SHAREHOLDER', 'BENEFICIAL_OWNER',
        'EMPLOYEE', 'CONSULTANT', 'AGENT', 'REPRESENTATIVE', 'SIGNATORY',
        
        # Financial & Legal
        'BANK', 'FINANCIAL_INSTITUTION', 'PAYMENT_PROCESSOR', 'EXCHANGE',
        'AMOUNT', 'CURRENCY', 'TRANSACTION', 'ACCOUNT', 'LICENSE', 'PERMIT',
        'CONTRACT', 'AGREEMENT', 'LAW', 'REGULATION', 'COMPLIANCE_REQUIREMENT',
        'REGULATOR', 'AUTHORITY', 'COURT', 'LEGAL_CASE',
        
        # Geographic & Temporal
        'COUNTRY', 'STATE', 'CITY', 'ADDRESS', 'JURISDICTION', 'TAX_HAVEN',
        'DATE', 'PERIOD', 'DEADLINE', 'FISCAL_YEAR',
        
        # Business Operations
        'PRODUCT', 'SERVICE', 'TECHNOLOGY', 'PATENT', 'TRADEMARK',
        'PROJECT', 'INITIATIVE', 'STRATEGY', 'MARKET', 'INDUSTRY',
        'CLIENT', 'CUSTOMER', 'SUPPLIER', 'VENDOR', 'PARTNER',
        
        # Other
        'OTHER'
    }
    
    validated = []  # Start with empty list of validated entities
    
    # Check each entity one by one
    for entity in entities:
        # Make sure the entity is a dictionary (object) with the right structure
        if not isinstance(entity, dict):
            logger.warning(f"Skipping non-dictionary entity: {entity}")
            continue  # Skip this entity and move to the next one
            
        # Make sure the entity has both required fields: 'name' and 'type'
        if 'name' not in entity or 'type' not in entity:
            logger.warning(f"Skipping entity missing required fields: {entity}")
            continue  # Skip this entity and move to the next one
        
        # Clean up the entity data
        name = str(entity['name']).strip()  # Convert to string and remove whitespace
        entity_type = str(entity['type']).strip().upper()  # Convert to uppercase string
        
        # Skip entities with empty names (not useful)
        if not name:
            logger.warning(f"Skipping entity with empty name: {entity}")
            continue
        
        # Check if the entity type is valid, if not change it to 'OTHER'
        if entity_type not in valid_types:
            logger.warning(f"Unknown entity type '{entity_type}', changing to 'OTHER'")
            entity_type = 'OTHER'
        
        # Skip very short names that are likely noise or errors
        if len(name) < 2:
            logger.warning(f"Skipping very short entity name: '{name}'")
            continue
        
        # If we get here, the entity is valid - add it to our validated list
        validated.append({
            'name': name,
            'type': entity_type
        })
    
    # Remove duplicate entities while preserving the order
    # We consider entities duplicate if they have the same name and type (case-insensitive)
    seen = set()  # Keep track of entities we've already seen
    deduplicated = []  # Final list without duplicates
    
    for entity in validated:
        # Create a unique key for this entity (lowercase name + type)
        entity_key = (entity['name'].lower(), entity['type'])
        if entity_key not in seen:  # If we haven't seen this entity before
            seen.add(entity_key)  # Mark it as seen
            deduplicated.append(entity)  # Add it to our final list
    
    # Log how many duplicates we removed (if any)
    if len(deduplicated) != len(validated):
        logger.info(f"Removed {len(validated) - len(deduplicated)} duplicate entities")
    
    return deduplicated  # Return the clean, validated, deduplicated entities

# =============================================================================
# DEVELOPER NOTES AND CONFIGURATION GUIDE
# =============================================================================

# WHY WE USE GROQ INSTEAD OF LOCAL LLAMA:
# 1. SPEED: Groq's inference is extremely fast (up to 10x faster than local models)
# 2. RELIABILITY: No need to manage local model installations or GPU memory
# 3. SCALABILITY: Can handle high-volume document processing without hardware limits
# 4. CONSISTENCY: Cloud-based models provide more consistent results across runs
# 5. MAINTENANCE: No need to update or manage local model files

# HOW TO SWITCH MODELS:
# To use a different Groq model, change the 'model' parameter in the API call:
# - "llama3-8b-8192": Llama 3 8B (fast, good for most tasks)
# - "llama3-70b-8192": Llama 3 70B (slower but more accurate)
# - "mixtral-8x7b-32768": Mixtral (good balance of speed and accuracy)
# - "openai/gpt-oss-120b": Current model (very large, high accuracy)

# WHAT JUNIOR DEVELOPERS SHOULD NOT CHANGE:
# 1. The function signature of extract_entities() - other code depends on it
# 2. The return format (List[Dict[str, str]]) - the pipeline expects this structure
# 3. The entity types in valid_types - these are specific to AML/KYC compliance
# 4. The _parse_llm_response() and _validate_entities() helper functions
# 5. The error handling structure - this prevents the pipeline from crashing

# SAFE TO MODIFY:
# 1. The prompt text (to improve entity extraction quality)
# 2. Temperature and max_tokens parameters (for different response styles)
# 3. Logging messages (for better debugging)
# 4. The model name (to try different Groq models)
# 5. Additional validation rules in _validate_entities()