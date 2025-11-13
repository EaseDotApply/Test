# November Member QA Service

A question-answering system that answers natural-language questions about member data from the November public API.

## Overview

This service provides a simple API endpoint `/ask` that accepts questions and returns answers based on member messages. The system uses:

- **FastAPI** for the REST API
- **LangChain** for retrieval-augmented generation (RAG)
- **Open-source models** (sentence-transformers for embeddings, mock LLM for generation)
- **Hybrid retrieval** combining FAISS vector search and BM25 lexical search

## API Endpoint

### POST `/api/ask`

**Request:**
```json
{
  "question": "When is Layla planning her trip to London?"
}
```

**Response:**
```json
{
  "answer": "Based on the messages, Layla is planning her trip to London in June."
}
```

## Example Questions

- "When is Layla planning her trip to London?"
- "How many cars does Vikram Desai have?"
- "What are Amira's favorite restaurants?"

## Design Alternatives Considered

When building this question-answering system, several architectural approaches were evaluated. Below are the key alternatives considered and the rationale for the chosen approach.

### 1. Hosted Closed-Source LLMs (e.g., OpenAI GPT-4o, Anthropic Claude)

**Approach**: Use commercial API services like OpenAI or Anthropic for both embeddings and text generation.

**Why Considered**:
- Superior performance and accuracy out-of-the-box
- No local model management or infrastructure requirements
- Built-in safety and moderation features
- Fast development cycle

**Why Rejected**:
- **Cost**: Per-request pricing can become expensive at scale
- **Dependency**: External API availability becomes a single point of failure
- **Latency**: Network round-trips add latency to every request
- **Data Privacy**: Sending member messages to third-party services raises privacy concerns
- **Requirement**: Assessment explicitly requires open-source stack

**Chosen Alternative**: Self-hosted open-source models (sentence-transformers for embeddings, Ollama-compatible LLMs for generation) provide cost control, data privacy, and independence from external services.

### 2. Structured SQL/Elasticsearch Backend

**Approach**: Store messages in a relational database (PostgreSQL) or search engine (Elasticsearch) with full-text search capabilities.

**Why Considered**:
- Mature ecosystem with proven scalability
- Rich query capabilities (filtering, aggregations, faceted search)
- ACID transactions for data consistency
- Industry-standard tooling and monitoring

**Why Rejected**:
- **Deployment Complexity**: Requires separate database service, connection pooling, and maintenance
- **Resource Overhead**: Additional infrastructure costs and operational burden
- **Over-engineering**: For a simple QA task, a full database is unnecessary
- **Semantic Search Limitations**: Traditional SQL/text search struggles with semantic similarity queries

**Chosen Alternative**: In-process FAISS (dense vector search) + BM25 (lexical search) hybrid retrieval. This provides:
- Self-contained deployment (no external services)
- Fast semantic search via embeddings
- Lexical matching for exact term queries
- Minimal infrastructure requirements

### 3. Agentic ReAct Pipeline with Tool Use

**Approach**: Use LangChain agents with ReAct (Reasoning + Acting) framework, allowing the LLM to use tools and make iterative decisions.

**Why Considered**:
- More flexible reasoning for complex multi-step questions
- Can dynamically choose retrieval strategies
- Better handling of ambiguous queries
- Enables iterative refinement of answers

**Why Rejected**:
- **Unpredictability**: Agent behavior can be non-deterministic, making testing and debugging difficult
- **Latency**: Multiple LLM calls per query increase response time
- **Complexity**: More moving parts increase failure modes
- **Cost**: Higher token usage with multiple reasoning steps

**Chosen Alternative**: Deterministic RAG (Retrieval-Augmented Generation) chain with explicit heuristics:
- Single-pass retrieval and generation
- Predictable behavior for consistent results
- Explicit numeric/date extraction rules for structured queries
- Faster and more cost-effective

### 4. Pure Dense Vector Search (FAISS only)

**Approach**: Use only semantic embeddings with FAISS, without lexical BM25 component.

**Why Considered**:
- Simpler architecture
- Semantic search handles synonyms and paraphrasing well
- Single retrieval method reduces complexity

**Why Rejected**:
- **Exact Match Failures**: Struggles with specific names, numbers, or exact phrases
- **Domain-Specific Terms**: May miss important lexical patterns in member messages
- **Lower Recall**: Some relevant documents may be missed without lexical matching

**Chosen Alternative**: Hybrid ensemble (60% FAISS, 40% BM25) combines the best of both:
- Semantic understanding from embeddings
- Exact term matching from BM25
- Improved recall and precision

### 5. Fine-Tuned Domain-Specific Model

**Approach**: Fine-tune a language model on the member messages dataset for better domain adaptation.

**Why Considered**:
- Potentially better understanding of domain-specific terminology
- Improved performance on member-specific queries
- Reduced need for prompt engineering

**Why Rejected**:
- **Data Requirements**: Requires substantial labeled training data
- **Training Time**: Significant computational resources and time
- **Maintenance**: Model must be retrained as new messages arrive
- **Overkill**: Pre-trained models with RAG perform well for this use case

**Chosen Alternative**: Pre-trained models (sentence-transformers, general-purpose LLMs) with RAG provide strong performance without training overhead.

## Data Insights

The system performs comprehensive analysis of the member messages dataset to identify data quality issues, patterns, and anomalies. This analysis helps ensure accurate question answering by flagging potential data inconsistencies.

### Dataset Overview

Based on analysis of the November member messages API:

- **Total Messages Analyzed**: 100 messages
- **Most Active Members**: 
  - Sophia Al-Farsi (16 messages)
  - Fatima El-Tahir (15 messages)
  - Hans Müller (11 messages)
- **Average Message Length**: 11.0 tokens per message

### Anomaly Detection

The system automatically detects and reports the following data quality issues:

#### 1. Duplicate Message IDs
**What it detects**: Messages with identical IDs appearing multiple times in the dataset.

**Why it matters**: Duplicate IDs indicate data ingestion errors or API pagination issues. This can lead to:
- Incorrect answer confidence scores
- Biased retrieval results (same message counted multiple times)
- Misleading statistics about message volume

**Detection method**: Checks for duplicate values in the `id` field across all fetched messages.

**Current status**: No duplicate IDs detected in the current dataset.

#### 2. Future Timestamps
**What it detects**: Messages with timestamps in the future relative to the current time.

**Why it matters**: Future timestamps indicate:
- Clock synchronization issues in the source system
- Data corruption or incorrect timezone handling
- Test data mixed with production data

**Impact**: Can cause incorrect temporal reasoning (e.g., "recent messages" queries may return future-dated messages).

**Detection method**: Compares message timestamps against current UTC time.

**Current status**: No future timestamps detected in the current dataset.

#### 3. Unusually Long Messages
**What it detects**: Messages exceeding the 99th percentile in token count.

**Why it matters**: Extremely long messages may:
- Indicate data corruption or concatenation errors
- Contain spam or noise that degrades retrieval quality
- Require special handling in the QA pipeline

**Detection method**: Calculates token count distribution and flags outliers above the 99th percentile threshold.

**Current status**: No unusually long messages detected in the current dataset.

#### 4. Empty Messages After Cleaning
**What it detects**: Messages that become empty after text preprocessing (whitespace removal, normalization, etc.).

**Why it matters**: Empty messages:
- Provide no useful information for question answering
- Waste storage and processing resources
- Can cause errors in downstream processing

**Detection method**: Checks message length after cleaning pipeline removes whitespace and normalizes text.

**Current status**: No empty messages detected after cleaning in the current dataset.

### Data Quality Summary

The current dataset shows **good data quality** with:
- ✅ No duplicate message IDs
- ✅ No future timestamps
- ✅ No unusually long messages
- ✅ No empty messages after cleaning

This indicates the November API provides clean, consistent data suitable for reliable question answering.

### Accessing Insights

Insights are available via:
- **API Endpoint**: `GET /api/insights` - Returns JSON with highlights and anomalies
- **Generated Report**: `reports/insights.md` - Markdown report written after each analysis

### Continuous Monitoring

The insights service runs automatically:
- When data is refreshed via `/api/refresh`
- When preprocessing is triggered via CLI
- Can be manually invoked via the insights endpoint

This ensures data quality issues are detected early and can be addressed before they impact answer accuracy.

## Technical Stack

- **Framework**: FastAPI (Python 3.12+)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Lexical Search**: BM25
- **LLM**: Mock LLM (can be replaced with Ollama or other OSS models)

## Project Structure

```
app/
  api/            # FastAPI routes and schemas
  clients/        # API client for fetching messages
  core/           # Application configuration and logging
  domain/         # Data models
  pipelines/      # Data preprocessing
  services/       # QA service, vectorstore, insights
tests/            # Test suite
```

## Installation & Setup

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed setup and running instructions.

## Testing

```bash
# Run all tests
pytest tests/

# Test the API endpoint
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

## Deployment

The service can be deployed to any platform supporting Python applications:

- **Local with tunnel**: Use `cloudflared tunnel --url http://localhost:8000`
- **Cloud platforms**: Deploy to Render, Fly.io, Railway, or similar
- **Docker**: Containerize using the provided Dockerfile (if created)

Ensure the service is publicly accessible for assessment submission.

---

Built for the November QA assessment. All code follows best practices with modular architecture, comprehensive error handling, and clean separation of concerns.
