# 🏦 UK Banking Regulatory Compliance Assistant

> An intelligent RAG-powered compliance assistant for UK banking 
> regulations, built with Claude AI, LangGraph, and Qdrant.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Claude](https://img.shields.io/badge/Claude-Sonnet-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue)

---

## 🎯 What It Does

A production-grade AI assistant that helps compliance officers, 
analysts, and bank staff navigate UK financial regulations including:

- **FCA Consumer Duty** requirements and obligations
- **AML/KYC** anti-money laundering procedures
- **Operational Resilience** FCA/PRA requirements
- **GDPR** data protection compliance
- **Document Compliance Checker** — paste any banking document 
  for instant regulatory risk analysis

---

## 🏗️ Architecture
User Query
│
▼
┌─────────────────────────────────────────┐
│           LangGraph Pipeline            │
│                                         │
│  Guardrail → Cache → Router             │
│                │                        │
│         ┌──────┴──────┐                 │
│         ▼             ▼                 │
│      Rewrite      Web Search            │
│         │                               │
│         ▼                               │
│       HyDE                              │
│         │                               │
│         ▼                               │
│    Hybrid Retrieve                      │
│    (Qdrant + BM25 + Reranker)          │
│         │                               │
│         ▼                               │
│      Generate (Claude)                  │
│         │                               │
│         ▼                               │
│  Compliance Check → Critique → Retry    │
└─────────────────────────────────────────┘
---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet (Anthropic) |
| Orchestration | LangGraph |
| Vector Database | Qdrant |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Retrieval | Hybrid (Vector + BM25 + FlashRank reranking) |
| Query Enhancement | HyDE + Query Rewriting |
| Frontend | Streamlit |
| API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |

---

## 🚀 Key Features

- **Hybrid RAG** — combines vector search + BM25 keyword search
- **HyDE** — Hypothetical Document Embedding for better retrieval
- **Query Rewriting** — reformulates questions for regulatory terminology
- **Self-Critique Loop** — evaluates and retries weak answers
- **Semantic Cache** — caches similar queries to reduce API costs
- **Document Compliance Checker** — analyses documents for violations
- **Conversation Memory** — maintains context across sessions
- **Guardrails** — blocks off-topic and harmful queries

---

## 📚 Knowledge Base

- FCA Consumer Duty (PS22/9)
- FCA Financial Crime Guide
- FCA Business Plan 2024-25
- Operational Resilience (CP19/32)
- Treating Customers Fairly (TCF)
- FSA Handbook

---

## 🏃 Running Locally

### Prerequisites
- Docker Desktop
- Anthropic API key

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/uk-banking-compliance-rag.git
cd uk-banking-compliance-rag

# Add your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Start both containers
docker-compose up --build

# Visit
http://localhost:8501
```

### Ingest Documents
```bash
# Add PDFs to documents/ folder then run
python ingestion.py
```

---

## 📁 Project Structure
uk-banking-compliance-rag/
├── app.py              # Streamlit frontend
├── graph.py            # LangGraph workflow
├── nodes.py            # Agent node functions
├── retriever.py        # Qdrant hybrid retrieval
├── ingestion.py        # Document ingestion pipeline
├── cache.py            # Semantic caching
├── history.py          # Conversation memory
├── config.py           # Configuration
├── Dockerfile          # App container
├── docker-compose.yml  # Multi-container setup
└── documents/          # FCA/PRA regulatory PDFs

---

## 👨‍💻 Author

**Daniel** — Senior Intelligence Analyst & AI Engineer  
MSc Data Science & Analytics  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## ⚠️ Disclaimer

This tool is for educational and demonstration purposes. 
Always consult qualified legal and compliance professionals 
for regulatory advice.
