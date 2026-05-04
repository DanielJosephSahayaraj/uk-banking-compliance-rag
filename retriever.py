import numpy as np
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    K_VECTOR, K_BM25, FINAL_K
)
import uuid

# ── Embeddings ──
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ── Reranker ──
reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# ── Qdrant Client ──
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def store_chunks(chunks: list[str], metadata: list[dict] = None):
    vectors = embedder.encode(chunks).tolist()
    points = []
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        payload = {"text": chunk}
        if metadata and i < len(metadata):
            payload.update(metadata[i])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload=payload
        ))
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} chunks in Qdrant.")

def hybrid_retrieve(query: str, rewrite_query: str, hyde_answer: str) -> list[str]:
    # Vector search using HyDE answer
    vector_results = vector_search(hyde_answer, k=K_VECTOR)

    # BM25 over vector results
    if vector_results:
        tokenized = [t.lower().split() for t in vector_results]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(rewrite_query.lower().split())
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K_BM25]
        bm25_results = [vector_results[i] for i in bm25_indices]
    else:
        bm25_results = []

    # Combine + deduplicate
    combined = list(dict.fromkeys(vector_results + bm25_results))
    combined_dicts = [{"text": t} for t in combined]

    # Rerank
    if combined_dicts:
        rerank_request = RerankRequest(query=query, passages=combined_dicts)
        reranked = reranker.rerank(rerank_request)
        return [item["text"] for item in reranked[:FINAL_K]]
    return []

def vector_search(query: str, k: int = K_VECTOR) -> list[str]:
    query_vec = embedder.encode([query])[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=k
    )
    return [r.payload["text"] for r in results]