import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0.3  # low = more factual for compliance

# ── Qdrant ──
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "banking_compliance"

# ── Embeddings ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Retrieval ──
K_VECTOR = 12
K_BM25 = 12
FINAL_K = 4

# ── Cache ──
MAX_CACHE_SIZE = 1000
MAX_AGE_DAYS = 7
CACHE_FILE = "rag_cache.pkl"
SIMILARITY_THRESHOLD = 0.85

# ── Banking Domain ──
ALLOWED_KEYWORDS = [
    "fca", "pra", "regulation", "compliance", "capital",
    "liquidity", "consumer duty", "mortgage", "basel",
    "aml", "kyc", "fraud", "financial crime", "conduct",
    "bank", "credit", "risk", "policy", "rulebook",
    "mcob", "bcobs", "cobs", "mifid", "gdpr", "psd2",
    "anti-money", "sanctions", "prudential", "buffer",
    "leverage", "stress", "test", "reporting", "disclosure"
]

BLOCKED_KEYWORDS = [
    "bomb", "hack", "violence", "racist", "sexist",
    "porn", "suicide", "illegal", "money laundering instructions"
]

# ── History ──
HISTORY_FILE = "chat_history.json"
MAX_HISTORY_MESSAGES = 10