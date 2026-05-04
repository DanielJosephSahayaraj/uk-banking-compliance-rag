import os
import re
import pickle
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from config import (
    CACHE_FILE, MAX_CACHE_SIZE, 
    MAX_AGE_DAYS, SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL
)

# ── Embedder for cache similarity ──
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ── Load cache from disk at startup ──
cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} cached items from disk")
    except Exception as e:
        print("Cache load failed (using empty cache):", e)


def normalize_query(q: str) -> str:
    """Clean and normalize query for consistent caching."""
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    q = ' '.join(q.split())
    return q


def get_cached_response(query: str) -> str | None:
    """Check if a similar query has been answered before."""
    norm_q = normalize_query(query)
    q_vec = np.array(embedder.encode(norm_q))

    now = datetime.utcnow()

    for cached_vec_tuple, cached_answer in list(cache.items()):
        cached_vec = np.array(cached_vec_tuple)

        # ── Cosine similarity ──
        sim = np.dot(q_vec, cached_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(cached_vec) + 1e-10
        )

        if sim < SIMILARITY_THRESHOLD:
            continue

        # ── Handle old cache format ──
        if isinstance(cached_answer, str):
            print(f"Migrating old cache entry sim={sim:.4f}")
            cache[cached_vec_tuple] = {
                "answer": cached_answer,
                "timestamp": now.isoformat()
            }
            return cached_answer

        # ── Handle new cache format ──
        if isinstance(cached_answer, dict):
            entry_time = datetime.fromisoformat(cached_answer["timestamp"])
            if now - entry_time <= timedelta(days=MAX_AGE_DAYS):
                print(f"Cache HIT! sim={sim:.4f}")
                return cached_answer["answer"]
            else:
                del cache[cached_vec_tuple]
                print("Expired cache entry removed")

    print("Cache MISS")
    return None


def save_to_cache(query: str, answer: str, rewrite_query: str = None):
    """Save a query-answer pair to cache."""
    q_lower = query.lower()

    # ── Skip time-sensitive queries ──
    skip_words = [
        "date", "time", "today", "now", "current",
        "news", "latest", "recent", "search", "web"
    ]
    if any(word in q_lower for word in skip_words):
        print("Time-sensitive query — not caching")
        return

    timestamp = datetime.utcnow().isoformat()
    entry = {"answer": answer, "timestamp": timestamp}

    # ── Cache original query ──
    norm_orig = normalize_query(query)
    q_vec_orig = embedder.encode(norm_orig)
    cache[tuple(q_vec_orig)] = entry

    # ── Also cache rewritten query if provided ──
    if rewrite_query:
        norm_rew = normalize_query(rewrite_query)
        q_vec_rew = embedder.encode(norm_rew)
        cache[tuple(q_vec_rew)] = entry

    # ── Evict oldest if cache full ──
    if len(cache) > MAX_CACHE_SIZE:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
        print("Cache full — removed oldest entry")

    # ── Persist to disk ──
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        print("Saved to persistent cache")
    except Exception as e:
        print("Cache save failed:", e)  