# Assessment Submission

## Overview

This project implements a question-answering system that answers natural-language questions about member data from the November public API.

## Requirements Met

### ✅ Core Requirements

1. **Question-Answering System**
   - ✅ Accepts natural-language questions
   - ✅ Answers based on member messages from the public API
   - ✅ Uses the GET /messages endpoint: https://november7-730026606190.europe-west1.run.app/docs

2. **API Endpoint**
   - ✅ Endpoint: `POST /api/ask`
   - ✅ Request format: `{"question": "..."}`
   - ✅ Response format: `{"answer": "..."}`

3. **Example Questions Supported**
   - ✅ "When is Layla planning her trip to London?"
   - ✅ "How many cars does Vikram Desai have?"
   - ✅ "What are Amira's favorite restaurants?"

4. **Deployment**
   - ⚠️ Service must be publicly accessible (see deployment instructions below)

### ✅ Bonus Goals

#### Bonus 1: Design Notes

**Location**: README.md, "Design Alternatives Considered" section

**Alternatives Considered:**

1. **Hosted Closed-Source LLMs (e.g., OpenAI GPT-4o)**
   - Rejected to honor requirement for open-source stack and eliminate external dependencies

2. **Structured SQL/Elasticsearch Backend**
   - Rejected in favor of in-process FAISS + BM25 for self-contained deployment

3. **Agentic ReAct Pipeline**
   - Rejected in favor of deterministic RAG chain with explicit heuristics for predictable behavior

#### Bonus 2: Data Insights

**Location**: README.md, "Data Insights" section

**Anomalies Detected:**

- Duplicate message IDs
- Future timestamps (data quality issues)
- Unusually long messages
- Empty messages after cleaning

**Access**: Available via `/api/insights` endpoint

## Technical Implementation

### Architecture

- **Framework**: FastAPI (Python 3.12+)
- **Retrieval**: Hybrid approach using FAISS (dense) + BM25 (lexical)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Generation**: Mock LLM (can be replaced with Ollama/OSS models)
- **Validation**: Transformers-based hallucination detection

### Key Features

1. **Hybrid Retrieval**: Combines semantic (FAISS) and lexical (BM25) search for robust recall
2. **Caching**: Messages cached locally in Parquet format for fast access
3. **Preprocessing**: Text normalization, deduplication, entity extraction
4. **Quality Guardrails**: Hallucination validation with confidence scoring

## API Usage

### Endpoint: POST /api/ask

**Request:**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

**Response:**
```json
{
  "answer": "Based on the messages, Layla is planning her trip to London in June."
}
```

### Additional Endpoints

- `GET /api/health` - Health check
- `GET /api/insights` - Data insights and anomalies
- `POST /api/refresh` - Refresh the search index

## Setup & Running

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for complete setup instructions.

**Quick Start:**
```bash
# Install
pip install -e .

# Fetch data and build index
python -m app.cli fetch
python -m app.cli build-index

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Testing

All tests pass:
```bash
pytest tests/
# Result: 5 passed, 1 skipped
```

## Deployment

### Option 1: Cloudflare Tunnel (Quick)

```bash
cloudflared tunnel --url http://localhost:8000
```

### Option 2: Cloud Platform

Deploy to Render, Fly.io, Railway, or similar platforms. See HOW_TO_RUN.md for details.

## Project Structure

```
app/
  api/            # FastAPI routes
  clients/        # API client
  core/           # Configuration
  domain/         # Data models
  pipelines/      # Preprocessing
  services/       # QA, vectorstore, insights
tests/            # Test suite
```

## Code Quality

- ✅ Modular architecture
- ✅ Type hints throughout
- ✅ Comprehensive error handling
- ✅ Structured logging
- ✅ Clean separation of concerns
- ✅ No linter errors

## Files

- `README.md` - Main project documentation
- `HOW_TO_RUN.md` - Setup and running instructions
- `ASSESSMENT.md` - This file (assessment submission details)
- `app/main.py` - Application entry point
- `app/api/routes.py` - API endpoint definitions

## Notes

- The service uses a mock LLM for testing. For production, configure Ollama or another OSS LLM.
- First request may be slow as models load into memory.
- Data is cached locally after first fetch for faster subsequent runs.

---

**Submission Checklist:**
- ✅ Code complete and tested
- ✅ README with design alternatives
- ✅ Data insights documented
- ⚠️ Public deployment (instructions provided)
- ⚠️ GitHub repository (to be created)

