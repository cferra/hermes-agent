"""
qdrant_memory.py — Semantic vector memory for Hermes using Qdrant + bge-m3

Collections:
  hermes_memory  — agent-saved facts/insights (replaces/augments memory_tool.py)
  hermes_docs    — chunked document RAG (optional, ingested separately)

Environment overrides:
  QDRANT_URL      — default: http://192.168.1.15:6333
  BGE_EMBED_URL   — default: http://192.168.1.33:9002/v1/embeddings
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.1.15:6333")
BGE_EMBED_URL = os.getenv("BGE_EMBED_URL", "http://192.168.1.33:9002/v1/embeddings")
MEMORY_COLLECTION = "hermes_memory"
DOCS_COLLECTION = "hermes_docs"
VECTOR_SIZE = 1024  # bge-m3 output dimension


def get_embedding(text: str) -> list[float]:
    """Call bge-m3 on intel-ai and return the 1024-dim embedding vector."""
    resp = requests.post(
        BGE_EMBED_URL,
        json={"model": "bge-m3", "input": text},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def ensure_collection(client: QdrantClient, collection: str = MEMORY_COLLECTION) -> None:
    """Create the collection if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", collection)


class QdrantMemoryStore:
    """
    Semantic vector memory backed by Qdrant + bge-m3 embeddings.

    Usage:
        store = QdrantMemoryStore()
        mem_id = store.upsert("User prefers dark mode", {"tags": ["preferences"]})
        results = store.search("what does the user like for UI?")
    """

    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, grpc_port=6334, timeout=10)
        ensure_collection(self.client, MEMORY_COLLECTION)

    def upsert(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Embed text and upsert into hermes_memory.
        Returns the point ID (UUID hex string).
        """
        mem_id = uuid.uuid4().hex
        vector = get_embedding(text)
        payload = {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        self.client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[PointStruct(id=mem_id, vector=vector, payload=payload)],
        )
        return mem_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.65,
        tag_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over hermes_memory.
        Returns list of {id, text, score, timestamp, ...metadata} dicts,
        sorted by descending relevance score.
        """
        vector = get_embedding(query)

        query_filter = None
        if tag_filter:
            query_filter = Filter(
                must=[FieldCondition(key="tags", match=MatchValue(value=tag_filter))]
            )

        response = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            entry = {"id": hit.id, "score": round(hit.score, 4)}
            entry.update(hit.payload or {})
            results.append(entry)
        return results

    def delete(self, memory_id: str) -> bool:
        """Delete a memory point by its ID. Returns True if successful."""
        try:
            self.client.delete(
                collection_name=MEMORY_COLLECTION,
                points_selector=PointIdsList(points=[memory_id]),
            )
            return True
        except Exception as e:
            logger.warning("Failed to delete memory %s: %s", memory_id, e)
            return False

    def list_recent(self, limit: int = 20) -> list[dict]:
        """Return the most recently added memories (scroll, no query vector needed)."""
        results, _ = self.client.scroll(
            collection_name=MEMORY_COLLECTION,
            limit=limit,
            with_payload=True,
        )
        entries = []
        for point in results:
            entry = {"id": point.id}
            entry.update(point.payload or {})
            entries.append(entry)
        # Sort by timestamp descending if available
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return entries[:limit]

    def collection_info(self) -> dict:
        """Return basic stats about the hermes_memory collection."""
        info = self.client.get_collection(MEMORY_COLLECTION)
        return {
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }
