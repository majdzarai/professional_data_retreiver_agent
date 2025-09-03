# 🔍 Graph RAG Research Pipeline
## Advanced KYC/AML Compliance Analysis System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.1-green.svg)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

A state-of-the-art **Graph RAG (Retrieval-Augmented Generation)** pipeline that transforms financial documents into actionable intelligence for KYC/AML compliance. This system leverages local AI models to extract entities, map relationships, and generate professional compliance reports without external API dependencies.

## 🎯 What This System Does

```
📄 Financial Documents → 🤖 AI Processing → 🕸️ Knowledge Graph → 📊 Intelligence Reports
```

### Core Capabilities
- **🔍 Entity Recognition**: Automatically identifies companies, people, locations, and financial instruments
- **🔗 Relationship Mapping**: Discovers business relationships, ownership structures, and transaction patterns
- **🕸️ Knowledge Graphs**: Builds comprehensive networks of interconnected entities
- **👥 Community Detection**: Identifies clusters of related entities for risk assessment
- **📋 Compliance Reports**: Generates professional KYC/AML analysis with risk ratings
- **🛡️ Privacy-First**: All processing done locally with no external API calls

## 🚀 Quick Start

### Prerequisites
- **Python 3.7+** with pip
- **Ollama** with Llama 3.1 model
- **8GB+ RAM** recommended for optimal performance

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/majdzarai/professional_data_retreiver_agent.git
   cd datapipeline
   ```

2. **Install Dependencies**
   ```bash
   pip install networkx requests python-louvain matplotlib seaborn pandas numpy PyPDF2
   ```

3. **Setup Ollama AI**
   ```bash
   # Install Ollama (visit https://ollama.ai for platform-specific instructions)
   ollama pull llama3.1:latest
   ollama serve
   ```

4. **Prepare Your Documents**
   ```bash
   mkdir -p input_data
   # Place your PDF or TXT files in input_data/
   ```

5. **Run the Pipeline**
   ```bash
   python main_researcher.py
   ```

## 📁 Project Architecture

```
datapipeline/
├── 🎯 main_researcher.py          # Main pipeline orchestrator
├── 📂 graph_rag/                  # Core AI modules
│   ├── 📄 pdf_extractor.py        # PDF text extraction
│   ├── 🔪 chunking.py             # Intelligent text segmentation
│   ├── 🏷️ entity_extraction.py    # AI-powered entity recognition
│   ├── 🔗 relationship_extraction.py # Relationship mapping
│   ├── 🕸️ graph_builder.py        # Knowledge graph construction
│   ├── 👥 community_detection.py  # Clustering algorithms
│   └── 📋 community_summary.py    # Intelligence report generation
├── 📂 input_data/                 # Your documents go here
├── 📂 output_data/                # Generated intelligence
│   ├── cleaned_data/              # Processed text
│   ├── chunks/                    # Text segments
│   ├── entities/                  # Extracted entities
│   ├── relationships/             # Entity relationships
│   ├── graph/                     # Knowledge graphs
│   ├── communities/               # Detected clusters
│   └── summaries/                 # Final reports
└── 📂 test/                       # Testing utilities
```

## 🔄 Pipeline Workflow

The Graph RAG pipeline processes documents through 7 intelligent stages:

### Stage 1: 📄 Document Processing
- Extracts clean text from PDF or TXT files
- Handles complex document layouts and formatting
- Preserves semantic structure for analysis

### Stage 2: 🔪 Intelligent Chunking
- Segments text into semantically meaningful chunks
- Maintains context boundaries for accurate analysis
- Optimizes chunk size for AI processing

### Stage 3: 🏷️ Entity Extraction
- Uses local Llama 3.1 to identify key entities:
  - **Companies**: Corporations, subsidiaries, partnerships
  - **People**: Executives, beneficial owners, signatories
  - **Locations**: Jurisdictions, addresses, tax havens
  - **Financial**: Accounts, transactions, instruments

### Stage 4: 🔗 Relationship Mapping
- Discovers semantic relationships between entities:
  - **Ownership**: Controls, owns, subsidiary_of
  - **Business**: Partners_with, transacts_with, supplies
  - **Legal**: Regulated_by, licensed_in, incorporated_in
  - **Financial**: Transfers_to, receives_from, guarantees

### Stage 5: 🕸️ Knowledge Graph Construction
- Builds comprehensive network of entities and relationships
- Validates data consistency and removes duplicates
- Creates graph structure optimized for analysis

### Stage 6: 👥 Community Detection
- Applies advanced clustering algorithms (Greedy Modularity)
- Identifies groups of closely related entities
- Calculates community metrics and connectivity scores

### Stage 7: 📋 Intelligence Report Generation
- Generates professional KYC/AML compliance reports
- Provides risk assessments and regulatory insights
- Creates both structured JSON and human-readable formats

## 📊 Output Examples

### Knowledge Graph Statistics
```
📈 Graph Statistics:
   • Total entities: 156
   • Total relationships: 89
   • Communities detected: 4
   • Processing time: 2.3 minutes
```

### Community Analysis Report
```
COMMUNITY 1 - RISK LEVEL: HIGH
═══════════════════════════════════════
Entity Count: 23 entities
Connectivity: Dense (0.78)

KYC/AML ASSESSMENT:
**THEME:** International Financial Services Network
**KEY ENTITIES:** GlobalBank Ltd, OffshoreHoldings Inc, TaxHaven Corp
**RISK FACTORS:**
- Multiple offshore jurisdictions
- Complex ownership structures
- High-risk geography exposure

**REGULATORY RECOMMENDATIONS:**
- Enhanced due diligence required
- Source of funds verification
- Ongoing monitoring protocols
```

### Generated Files
After pipeline completion, you'll find:

- **📄 Text Data**: `output_data/cleaned_data/document_cleaned.txt`
- **🔪 Chunks**: `output_data/chunks/document_chunks.txt`
- **🏷️ Entities**: `output_data/entities/document_entities.json`
- **🔗 Relationships**: `output_data/relationships/document_relationships.json`
- **🕸️ Graph**: `output_data/graph/document_graph.json`
- **👥 Communities**: `output_data/communities/document_communities.json`
- **📋 Reports**: `output_data/summaries/document_summaries_report.txt`

## ⚙️ Configuration

### Input File Setup
Edit the input file path in `main_researcher.py`:
```python
# Default input file path (modify as needed)
input_file = "input_data/your_document.pdf"
```

### AI Model Configuration
The pipeline uses Llama 3.1 by default. To use different models, update the extraction modules:
```python
# In entity_extraction.py and relationship_extraction.py
LOCAL_LLM_MODEL = "llama3.1:latest"  # Change to your preferred model
```

### Processing Limits
For testing, the pipeline processes limited chunks. For production:
```python
# In main_researcher.py, change:
test_chunks = chunks[:1]  # Testing: first chunk only
# To:
test_chunks = chunks      # Production: all chunks
```

## 🛡️ Security & Compliance

### Privacy Protection
- **Local Processing**: All AI operations run on your machine
- **No External APIs**: Zero data transmission to third parties
- **Secure Storage**: All outputs saved locally with proper permissions

### Compliance Features
- **AML Screening**: Identifies potential money laundering indicators
- **KYC Analysis**: Maps beneficial ownership and control structures
- **Risk Assessment**: Automated risk scoring based on entity patterns
- **Audit Trail**: Complete processing logs for regulatory review

## 🔧 Troubleshooting

### Common Issues

**❌ "Cannot connect to Ollama server"**
```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

**❌ "No Llama 3.1 models found"**
```bash
# Install the model
ollama pull llama3.1:latest

# Verify installation
ollama list
```

**❌ "Input file not found"**
- Ensure your document is in the `input_data/` directory
- Update the file path in `main_researcher.py`
- Check file permissions and encoding

**❌ "Memory errors during processing"**
- Reduce chunk size in `chunking.py`
- Process fewer chunks at once
- Increase system RAM or use smaller models

## 🧪 Testing

Run individual pipeline components:

```bash
# Test entity extraction
python test/test_entity_pipeline.py

# Test graph construction
python test/test_graph_builder.py

# Test community detection
python test/test_communities.py

# Test LLM connectivity
python test/test_local_llama.py
```

## 🚀 Performance Optimization

### For Large Documents
- **Parallel Processing**: Modify chunking to process multiple chunks simultaneously
- **Batch Operations**: Group entity/relationship extractions for efficiency
- **Memory Management**: Implement streaming for very large files

### For Production Use
- **Database Integration**: Store graphs in Neo4j or similar graph databases
- **Caching**: Cache entity/relationship extractions to avoid reprocessing
- **Monitoring**: Add performance metrics and health checks

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **📧 Issues**: [GitHub Issues](https://github.com/majdzarai/professional_data_retreiver_agent/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/majdzarai/professional_data_retreiver_agent/discussions)
- **📖 Documentation**: This README and inline code comments

## 🔮 Roadmap

### Upcoming Features
- [ ] **Real-time Processing**: Stream processing for live document feeds
- [ ] **Multi-language Support**: Process documents in multiple languages
- [ ] **Advanced Visualization**: Interactive graph exploration interface
- [ ] **API Endpoints**: REST API for integration with other systems
- [ ] **Enhanced ML Models**: Custom fine-tuned models for financial documents
- [ ] **Regulatory Templates**: Pre-built templates for different jurisdictions

### Performance Improvements
- [ ] **GPU Acceleration**: CUDA support for faster AI processing
- [ ] **Distributed Processing**: Multi-node processing for enterprise scale
- [ ] **Incremental Updates**: Process only changed documents

---

## ⚖️ Legal Notice

**This tool is designed for legitimate KYC/AML compliance purposes only.**

- Ensure proper authorization before analyzing business documents
- Comply with local data protection and privacy regulations
- Use responsibly and in accordance with applicable laws
- The authors are not responsible for misuse of this software

---

**Built with ❤️ for the compliance community**

*Empowering financial institutions with AI-driven intelligence while maintaining privacy and security.*