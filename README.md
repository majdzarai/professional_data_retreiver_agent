# KYC/AML Graph RAG Pipeline

A comprehensive Knowledge Graph-based Retrieval Augmented Generation (RAG) pipeline designed for KYC (Know Your Customer) and AML (Anti-Money Laundering) analysis. This system extracts entities and relationships from business documents, builds knowledge graphs, and performs community detection to identify business clusters and risk patterns.

## 🚀 Features

- **Entity Extraction**: Automated extraction of business entities (companies, people, locations, etc.) using local LLM
- **Relationship Extraction**: Identification of business relationships and connections
- **Knowledge Graph Construction**: Building comprehensive knowledge graphs from extracted data
- **Community Detection**: Advanced clustering algorithms to identify business communities
- **KYC/AML Analysis**: Professional risk assessment and compliance reporting
- **LLM Integration**: Local Ollama integration for intelligent analysis and summarization

## 📋 Prerequisites

- Python 3.7+
- Ollama with llama3.1 model installed
- Required Python packages (see requirements below)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/majdzarai/professional_data_retreiver_agent.git
cd professional_data_retreiver_agent
```

2. Install required packages:
```bash
pip install networkx requests python-louvain matplotlib seaborn pandas numpy
```

3. Install and start Ollama:
```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
ollama pull llama3.1:latest
ollama serve
```

## 📁 Project Structure

```
datapipeline/
├── graph_rag/                 # Core pipeline modules
│   ├── entity_extraction.py   # Entity extraction using LLM
│   ├── relationship_extraction.py # Relationship extraction
│   ├── graph_builder.py       # Knowledge graph construction
│   ├── community_detection.py # Community detection algorithms
│   ├── community_summary.py   # KYC/AML analysis and reporting
│   ├── chunking.py            # Text preprocessing
│   └── pdf_extractor.py       # PDF text extraction
├── test/                      # Test scripts and examples
│   ├── test_entity_pipeline.py # Entity extraction testing
│   ├── test_graph_builder.py  # Graph construction testing
│   ├── test_communities.py    # Community detection testing
│   ├── test_local_llama.py    # LLM connectivity testing
│   └── visualize_graph.py     # Graph visualization
├── input_data/                # Input documents
├── output_data/               # Generated outputs
└── README.md
```

## 🔄 Pipeline Workflow

### 1. Entity and Relationship Extraction
```bash
python test/test_entity_pipeline.py
```
Extracts entities and relationships from business documents using local LLM.

### 2. Knowledge Graph Construction
```bash
python test/test_graph_builder.py
```
Builds a comprehensive knowledge graph from extracted data.

### 3. Community Detection and KYC/AML Analysis
```bash
python test/test_communities.py
```
Performs community detection and generates professional KYC/AML reports.

## 📊 Output Files

- **Entities**: `test/entities_output_YYYY-MM-DD_HH-MM-SS.json`
- **Relationships**: `test/relations_output_YYYY-MM-DD_HH-MM-SS.json`
- **Knowledge Graph**: `test/knowledge_graph_YYYY-MM-DD_HH-MM-SS.json`
- **Community Analysis**: `test/summaries.json`
- **KYC/AML Report**: `test/summaries_report.txt`

## 🎯 KYC/AML Features

### Risk Assessment
- Automated entity classification and risk scoring
- Business relationship analysis
- Regulatory compliance checking
- Sanctions screening indicators

### Community Detection
- Greedy modularity-based clustering
- Rule-based business categorization
- Risk level assignment (High/Medium/Low)
- Network connectivity analysis

### Professional Reporting
- Executive summaries with risk metrics
- Detailed cluster analysis
- Compliance recommendations
- Machine-readable JSON outputs

## 🔧 Configuration

### LLM Settings
Edit `graph_rag/community_summary.py` to configure LLM parameters:
```python
LOCAL_LLM_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "llama3.1:latest"
```

### Risk Thresholds
Modify risk assessment criteria in `graph_rag/community_detection.py`:
```python
# Adjust cluster size and connectivity thresholds
HIGH_RISK_THRESHOLD = 50
MEDIUM_RISK_THRESHOLD = 20
```

## 📈 Example Output

### Knowledge Graph Statistics
```
📈 Graph Statistics:
   • Total nodes: 231
   • Total edges: 114
   • Most connected entities: IBA, Financial Services, Regulatory Bodies
```

### Community Analysis
```
CLUSTER 0 - RISK LEVEL: HIGH
Entity Count: 101 entities
KYC/AML ASSESSMENT:
**THEME:** Global Operations in Medical Technology Sector
**RISK ASSESSMENT:** Potential regulatory compliance requirements...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Include error logs and system information

## 🔮 Future Enhancements

- [ ] Neo4j database integration
- [ ] Real-time streaming analysis
- [ ] Advanced visualization dashboard
- [ ] Multi-language document support
- [ ] Enhanced ML-based risk scoring
- [ ] API endpoints for integration

---

**Note**: This tool is designed for legitimate KYC/AML compliance purposes. Ensure proper authorization before analyzing any business documents.