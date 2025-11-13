# How to Run the November QA Service

This guide will help you set up and run the question-answering service.

## Prerequisites

- Python 3.12 or higher
- pip package manager
- Internet connection (for downloading models and fetching data)

## Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

This will install all required packages including:
- FastAPI and Uvicorn
- LangChain and related libraries
- sentence-transformers (for embeddings)
- FAISS (for vector search)
- transformers (for hallucination detection)

## Step 2: Configure Environment (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` if you need to customize:
- API URL (defaults to November public API)
- Embedding model
- LLM settings

## Step 3: Fetch and Process Data

```bash
# Fetch messages from the API
python -m app.cli fetch

# Preprocess the messages
python -m app.cli preprocess

# Build the search index
python -m app.cli build-index
```

**Note**: The first run will download the embedding model (~80MB), which may take a few minutes.

## Step 4: Start the API Server

```bash
# Option 1: Using the CLI
python -m app.cli serve

# Option 2: Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Option 3: Using Make
make run
```

The service will start on `http://localhost:8000`

## Step 5: Test the Service

### Using curl

```bash
# Simple question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'

# Expected response:
# {"answer": "Based on the messages, Layla is planning her trip to London in June."}
```

### Using the CLI

```bash
python -m app.cli ask "How many cars does Vikram Desai have?"
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ask",
    json={"question": "What are Amira's favorite restaurants?"}
)
print(response.json())
# {"answer": "..."}
```

## Step 6: Access API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Making the Service Publicly Accessible

### Option 1: Cloudflare Tunnel (Quick Testing)

```bash
# Install cloudflared if not already installed
# macOS: brew install cloudflared
# Then run:
cloudflared tunnel --url http://localhost:8000
```

This will give you a public URL like `https://xxxxx.trycloudflare.com`

### Option 2: Deploy to Cloud Platform

**Render.com:**
1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -e .`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Fly.io:**
1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Deploy: `fly deploy`

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Make sure you activated the virtual environment and installed dependencies:
```bash
source .venv/bin/activate
pip install -e .
```

### Issue: "Cannot connect to Ollama"

**Solution**: The service uses a mock LLM by default. If you want to use Ollama:
1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull mistral:instruct`
3. Update `.env` with `QA_LLM_API_BASE=http://localhost:11434`

### Issue: Slow first request

**Solution**: The first request loads models into memory. Subsequent requests will be faster.

### Issue: "No messages found"

**Solution**: Make sure you ran the fetch and build-index commands:
```bash
python -m app.cli fetch
python -m app.cli build-index
```

## Available CLI Commands

```bash
# Fetch messages from API
python -m app.cli fetch [--force]

# Preprocess messages
python -m app.cli preprocess [--force]

# Build search index
python -m app.cli build-index [--force]

# Ask a question
python -m app.cli ask "your question here"

# Start the server
python -m app.cli serve [--reload]

# Run tests
pytest tests/
```

## Project Structure

- `app/main.py` - Application entry point
- `app/api/routes.py` - API endpoint definitions
- `app/services/qa.py` - Question answering logic
- `app/services/vectorstore.py` - Search index management
- `data/` - Cached messages and processed data
- `vectorstore/` - FAISS index files

## Next Steps

1. Test with the example questions from the assessment
2. Deploy to a public platform
3. Share the public URL for assessment submission

For questions or issues, check the main README.md or review the code comments.

