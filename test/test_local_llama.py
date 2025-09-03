#!/usr/bin/env python3
"""
Test entity and relationship extraction using local Llama 3.1 model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime
from graph_rag.pdf_extractor import process_all_files_in_folders
from graph_rag.chunking import chunk_text
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_local_llama(prompt, model="llama3.1"):
    """
    Call local Llama model using ollama command line
    """
    try:
        # Use ollama command line to call local model
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120,  # 2 minute timeout
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.error(f"Ollama error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Ollama call timed out")
        return None
    except Exception as e:
        logger.error(f"Error calling local Llama: {e}")
        return None

def extract_entities_local(text_chunk):
    """
    Extract entities using local Llama 3.1
    """
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

    logger.info(f"Extracting entities from chunk (length: {len(text_chunk)} chars)")
    response = call_local_llama(prompt)
    
    if not response:
        return []
    
    # Parse JSON response
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response, re.S)
        if json_match:
            json_content = json_match.group(0)
            
            # Clean common JSON formatting issues
            json_content = json_content.replace('\n', ' ')
            json_content = re.sub(r'\s+', ' ', json_content)
            json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
            json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas
            json_content = re.sub(r'([^,\[\{])\s*\{', r'\1,{', json_content)  # Add missing commas between objects
            
            entities = json.loads(json_content)
            logger.info(f"Extracted {len(entities)} entities")
            return entities
        else:
            logger.warning("No JSON found in response")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing entities JSON: {e}")
        # Try to extract individual objects if array parsing fails
        try:
            import re
            objects = re.findall(r'\{[^{}]*"name"[^{}]*"type"[^{}]*\}', response)
            entities = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if 'name' in obj and 'type' in obj:
                        entities.append(obj)
                except:
                    continue
            if entities:
                logger.info(f"Recovered {len(entities)} entities from malformed JSON")
                return entities
        except:
            pass
        return []
    except Exception as e:
        logger.error(f"Error parsing entities JSON: {e}")
        return []

def extract_relationships_local(text_chunk):
    """
    Extract relationships using local Llama 3.1
    """
    prompt = f"""You are an expert in AML/KYC compliance and financial due diligence. Extract ALL semantic relationships from the following text for comprehensive regulatory reporting and risk assessment.

Extract these relationship types with maximum coverage:

**OWNERSHIP & CONTROL:**
- OWNS, CONTROLS, MANAGES, OPERATES, GOVERNS
- SUBSIDIARY_OF, PARENT_OF, DIVISION_OF, BRANCH_OF
- ACQUIRED, MERGED_WITH, SPUN_OFF, DIVESTED
- HOLDS_STAKE_IN, INVESTED_IN, FUNDED, FINANCED

**EMPLOYMENT & GOVERNANCE:**
- EMPLOYS, HIRED, APPOINTED, PROMOTED, TERMINATED
- CEO_OF, CFO_OF, DIRECTOR_OF, PRESIDENT_OF
- BOARD_MEMBER_OF, ADVISOR_TO, CONSULTANT_FOR
- REPORTS_TO, SUPERVISES, OVERSEES, LEADS

**BUSINESS RELATIONSHIPS:**
- PARTNERS_WITH, COLLABORATES_WITH, JOINT_VENTURE
- SUPPLIES_TO, PROVIDES_TO, SERVICES, CONTRACTS_WITH
- COMPETES_WITH, RIVALS, CHALLENGES, THREATENS
- CUSTOMERS_INCLUDE, CLIENTS_ARE, SERVES, SELLS_TO

**FINANCIAL RELATIONSHIPS:**
- PAID, RECEIVED, TRANSFERRED, DEPOSITED, WITHDREW
- FINED, PENALIZED, SANCTIONED, CHARGED, SUED
- BORROWED_FROM, LENT_TO, CREDIT_FROM, DEBT_TO
- REVENUE_FROM, PROFIT_FROM, LOSS_FROM, COST_OF

**REGULATORY & LEGAL:**
- REGULATED_BY, SUPERVISED_BY, OVERSEEN_BY
- LICENSED_BY, AUTHORIZED_BY, APPROVED_BY
- INVESTIGATED_BY, AUDITED_BY, EXAMINED_BY
- COMPLIES_WITH, VIOLATES, BREACHES, FOLLOWS

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

Return ONLY a valid JSON array of objects. Each object must have exactly three fields: "source", "relation", and "target".

JSON:"""

    logger.info(f"Extracting relationships from chunk (length: {len(text_chunk)} chars)")
    response = call_local_llama(prompt)
    
    if not response:
        return []
    
    # Parse JSON response
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response, re.S)
        if json_match:
            json_content = json_match.group(0)
            
            # Clean common JSON formatting issues
            json_content = json_content.replace('\n', ' ')
            json_content = re.sub(r'\s+', ' ', json_content)
            json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
            json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas
            json_content = re.sub(r'([^,\[\{])\s*\{', r'\1,{', json_content)  # Add missing commas between objects
            
            relationships = json.loads(json_content)
            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
        else:
            logger.warning("No JSON found in response")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing relationships JSON: {e}")
        # Try to extract individual objects if array parsing fails
        try:
            import re
            objects = re.findall(r'\{[^{}]*"source"[^{}]*"relation"[^{}]*"target"[^{}]*\}', response)
            relationships = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if 'source' in obj and 'relation' in obj and 'target' in obj:
                        relationships.append(obj)
                except:
                    continue
            if relationships:
                logger.info(f"Recovered {len(relationships)} relationships from malformed JSON")
                return relationships
        except:
            pass
        return []
    except Exception as e:
        logger.error(f"Error parsing relationships JSON: {e}")
        return []

def test_local_llama_pipeline():
    """
    Test the complete pipeline using local Llama 3.1
    """
    print("🚀 Starting Local Llama 3.1 Entity & Relationship Extraction Test")
    print("=" * 70)
    
    # Check if ollama is available
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "llama3.1" not in result.stdout:
            print("❌ Error: llama3.1 model not found in ollama")
            print("Please run: ollama pull llama3.1")
            return
        else:
            print("✅ Local Llama 3.1 model found")
    except Exception as e:
        print(f"❌ Error: ollama not found or not running: {e}")
        print("Please install and start ollama first")
        return
    
    # Step 1: Extract and clean text
    print("\n📄 Step 1: Processing documents...")
    process_all_files_in_folders("pdf_data", "input_data", "output_data/cleaned_data")
    
    # Read processed files
    cleaned_data_dir = "output_data/cleaned_data"
    if not os.path.exists(cleaned_data_dir):
        print("❌ No cleaned data directory found")
        return
    
    documents = []
    for filename in os.listdir(cleaned_data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(cleaned_data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    if not documents:
        print("❌ No documents found to process")
        return
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Step 2: Chunk the text
    print("\n🔪 Step 2: Chunking text...")
    all_chunks = []
    for i, content in enumerate(documents):
        chunks = chunk_text(content)
        all_chunks.extend(chunks)
        print(f"📄 Document {i+1}:")
        print(f"   → {len(chunks)} chunks created")
    
    # Test with 5 chunks
    chunks_to_process = min(5, len(all_chunks))
    print(f"\n🔎 Step 3: Processing first {chunks_to_process} chunks with local Llama 3.1...")
    
    # Save chunks to file first
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chunks_file = f"test/chunks_output_{timestamp}.txt"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        f.write(f"=== CHUNKS OUTPUT ({chunks_to_process} chunks) ===\n\n")
        for i, chunk in enumerate(all_chunks[:chunks_to_process]):
            f.write(f"--- CHUNK {i+1} ---\n")
            f.write(f"Metadata: {chunk['metadata']}\n")
            f.write(chunk['text'])
            f.write(f"\n\n--- END CHUNK {i+1} ---\n\n")
    
    print(f"   📄 Chunks saved to: {chunks_file}")
    
    all_entities = []
    all_relationships = []
    
    for i, chunk in enumerate(all_chunks[:chunks_to_process]):
        print(f"\n   🔎 Chunk {i+1}/{chunks_to_process}")
        
        # Extract entities
        entities = extract_entities_local(chunk['text'])
        all_entities.extend(entities)
        
        # Extract relationships
        relationships = extract_relationships_local(chunk['text'])
        all_relationships.extend(relationships)
        
        print(f"      → {len(entities)} entities, {len(relationships)} relationships")
    
    # Save results to separate files
    entities_file = f"test/entities_output_{timestamp}.json"
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, indent=2, ensure_ascii=False)
    
    relations_file = f"test/relations_output_{timestamp}.json"
    with open(relations_file, 'w', encoding='utf-8') as f:
        json.dump(all_relationships, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Local Llama 3.1 Extraction Complete")
    print(f"   📄 Chunks saved to: {chunks_file}")
    print(f"   👥 Entities saved to: {entities_file}")
    print(f"   🔗 Relations saved to: {relations_file}")
    print(f"   Total chunks processed: {chunks_to_process}")
    print(f"   Total entities: {len(all_entities)}")
    print(f"   Total relationships: {len(all_relationships)}")
    
    # Show sample results
    if all_entities:
        print("\n📊 Sample Entities:")
        for entity in all_entities[:5]:
            print(f"   • {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
    
    if all_relationships:
        print("\n🔗 Sample Relationships:")
        for rel in all_relationships[:5]:
            print(f"   • {rel.get('source', 'N/A')} → {rel.get('relation', 'N/A')} → {rel.get('target', 'N/A')}")

if __name__ == "__main__":
    test_local_llama_pipeline()