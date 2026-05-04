FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install packages in stages to avoid conflicts
RUN pip install --no-cache-dir \
    anthropic \
    langchain-anthropic \
    langchain-community \
    langchain-text-splitters \
    langgraph

RUN pip install --no-cache-dir \
    qdrant-client \
    sentence-transformers \
    flashrank \
    rank-bm25

RUN pip install --no-cache-dir \
    streamlit \
    fastapi \
    uvicorn \
    pypdf \
    python-dotenv \
    httpx \
    pytest

# Copy application code
COPY . .

# Create documents directory
RUN mkdir -p documents

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]