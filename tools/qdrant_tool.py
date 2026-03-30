"""
qdrant_tool.py — Semantic vector memory tools for Hermes Agent

Provides two tools:
  qdrant_save   — embed and persist a fact/insight to long-term semantic memory
  qdrant_search — semantic (fuzzy/conceptual) search over saved memories

Backed by Qdrant on FERRANTEAI (192.168.1.15:6333) + bge-m3 embeddings on
intel-ai (192.168.1.33:9002). Use these alongside the keyword-based `memory`
tool — qdrant_search excels at conceptual queries where keyword matching fails.

Environment overrides:
  QDRANT_URL       — Qdrant REST endpoint (default: http://192.168.1.15:6333)
  BGE_EMBED_URL    — bge-m3 embeddings endpoint (default: http://192.168.1.33:9002/v1/embeddings)
  REDIS_STACK_URL  — Redis Stack for hot cache (default: redis://192.168.1.15:6380)
  QDRANT_CACHE_TTL — Search result cache TTL in seconds (default: 600)
"""

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis hot cache (redis-stack on 192.168.1.15:6380)
# Caches qdrant_search and qdrant_doc_search results for 10 minutes to avoid
# redundant embedding + Qdrant round-trips for repeated or identical queries.
# Falls back silently if redis-stack is unreachable.
# ---------------------------------------------------------------------------
_REDIS_CACHE_TTL = int(os.getenv('QDRANT_CACHE_TTL', '600'))
_REDIS_STACK_URL = os.getenv('REDIS_STACK_URL', 'redis://192.168.1.15:6380')
_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        _redis_client = redis.from_url(_REDIS_STACK_URL, decode_responses=True, socket_connect_timeout=2)
        _redis_client.ping()
    except Exception:
        _redis_client = None
    return _redis_client

def _cache_key(namespace: str, **kwargs) -> str:
    payload = json.dumps(kwargs, sort_keys=True)
    return f'qdrant:{namespace}:{hashlib.sha256(payload.encode()).hexdigest()[:16]}'

def _cache_get(key: str):
    try:
        r = _get_redis()
        if r:
            return r.get(key)
    except Exception:
        pass
    return None

def _cache_set(key: str, value: str):
    try:
        r = _get_redis()
        if r:
            r.setex(key, _REDIS_CACHE_TTL, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

QDRANT_SAVE_SCHEMA = {
    "name": "qdrant_save",
    "description": (
        "Save a fact, insight, or observation to long-term semantic vector memory. "
        "Use this for information you want to retrieve later by meaning/concept, "
        "not just by keyword. Good for: user preferences, cluster facts, learned "
        "behaviours, project-specific knowledge. "
        "Returns the memory ID on success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The fact or insight to save. Be specific and self-contained — this text will be retrieved verbatim.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of category tags (e.g. ['hardware', 'cluster'] or ['preferences', 'ui']).",
            },
        },
        "required": ["text"],
    },
}

QDRANT_SEARCH_SCHEMA = {
    "name": "qdrant_search",
    "description": (
        "Semantic search over long-term vector memory. "
        "Unlike keyword search, this finds conceptually related memories even when "
        "exact words don't match. Use this when: the user asks something you should "
        "know but don't have in the current session, when keyword memory search "
        "returns nothing, or when the query is conceptual/fuzzy. "
        "Returns up to 5 relevant memories with relevance scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query describing what you're looking for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
                "default": 5,
            },
            "score_threshold": {
                "type": "number",
                "description": "Minimum relevance score 0.0-1.0 (default 0.65). Lower = broader results.",
                "default": 0.65,
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_qdrant_save(args: dict, **kwargs) -> str:
    text = args.get("text", "").strip()
    if not text:
        return "Error: 'text' is required and cannot be empty."
    tags = args.get("tags") or []

    try:
        from tools.qdrant_memory import QdrantMemoryStore
        store = QdrantMemoryStore()
        mem_id = store.upsert(text, {"tags": tags})
        tag_str = f" [tags: {', '.join(tags)}]" if tags else ""
        return f"Saved to vector memory{tag_str}: {mem_id}"
    except Exception as e:
        logger.error("qdrant_save failed: %s", e)
        return f"Error saving to vector memory: {e}"


def _handle_qdrant_search(args: dict, **kwargs) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: 'query' is required."
    top_k = min(int(args.get("top_k", 5)), 10)
    score_threshold = float(args.get("score_threshold", 0.65))

    cache_key = _cache_key("search", query=query, top_k=top_k, score_threshold=score_threshold)
    cached = _cache_get(cache_key)
    if cached:
        logger.debug("qdrant_search cache hit: %s", cache_key)
        return cached

    try:
        from tools.qdrant_memory import QdrantMemoryStore
        store = QdrantMemoryStore()
        results = store.search(query, top_k=top_k, score_threshold=score_threshold)

        if not results:
            result = "No relevant memories found above the relevance threshold."
        else:
            lines = [f"Found {len(results)} relevant memor{'y' if len(results)==1 else 'ies'}:\n"]
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                text = r.get("text", "")
                tags = r.get("tags", [])
                ts = r.get("timestamp", "")[:19] if r.get("timestamp") else ""
                tag_str = f"  [tags: {', '.join(tags)}]" if tags else ""
                ts_str = f"  [{ts}]" if ts else ""
                lines.append(f"{i}. [{score:.3f}] {text}{tag_str}{ts_str}")
            result = "\n".join(lines)

        _cache_set(cache_key, result)
        return result
    except Exception as e:
        logger.error("qdrant_search failed: %s", e)
        return f"Error searching vector memory: {e}"


QDRANT_DOC_SEARCH_SCHEMA = {
    "name": "qdrant_doc_search",
    "description": (
        "Semantic search over indexed documents (FerranteAI cluster setup, "
        "project notes, ingested files). Use this when you need to recall "
        "specific technical details, configs, or procedures that were documented "
        "but aren't in your current memory. Returns the most relevant passages "
        "with source file and chunk context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you're looking for — use natural language, not keywords.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of passages to return (default 4, max 8).",
                "default": 4,
            },
            "score_threshold": {
                "type": "number",
                "description": "Minimum relevance score 0.0-1.0 (default 0.60).",
                "default": 0.60,
            },
        },
        "required": ["query"],
    },
}


def _handle_qdrant_doc_search(args: dict, **kwargs) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: 'query' is required."
    top_k = min(int(args.get("top_k", 4)), 8)
    score_threshold = float(args.get("score_threshold", 0.60))

    cache_key = _cache_key("docs", query=query, top_k=top_k, score_threshold=score_threshold)
    cached = _cache_get(cache_key)
    if cached:
        logger.debug("qdrant_doc_search cache hit: %s", cache_key)
        return cached

    try:
        from tools.qdrant_memory import get_embedding, QDRANT_URL
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, grpc_port=6334, timeout=10)
        vector = get_embedding(query)

        response = client.query_points(
            collection_name="hermes_docs",
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        if not response.points:
            result = "No relevant document passages found above the relevance threshold."
        else:
            lines = [f"Found {len(response.points)} relevant passage(s):\n"]
            for i, hit in enumerate(response.points, 1):
                p = hit.payload or {}
                score = round(hit.score, 3)
                filename = p.get("filename", "unknown")
                chunk_idx = p.get("chunk_index", "?")
                total = p.get("total_chunks", "?")
                text = p.get("text", "")
                lines.append(
                    f"--- [{score:.3f}] {filename} (chunk {chunk_idx}/{total}) ---\n{text}"
                )
            result = "\n\n".join(lines)

        _cache_set(cache_key, result)
        return result
    except Exception as e:
        logger.error("qdrant_doc_search failed: %s", e)
        return f"Error searching documents: {e}"


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_qdrant_available() -> bool:
    """Returns True if Qdrant is reachable. Used by registry to gate tool availability."""
    import requests
    qdrant_url = os.getenv("QDRANT_URL", "http://192.168.1.15:6333")
    try:
        resp = requests.get(f"{qdrant_url}/healthz", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="qdrant_save",
    toolset="memory",
    schema=QDRANT_SAVE_SCHEMA,
    handler=_handle_qdrant_save,
    check_fn=_check_qdrant_available,
    emoji="🔮",
)

registry.register(
    name="qdrant_search",
    toolset="memory",
    schema=QDRANT_SEARCH_SCHEMA,
    handler=_handle_qdrant_search,
    check_fn=_check_qdrant_available,
    emoji="🔍",
)

registry.register(
    name="qdrant_doc_search",
    toolset="memory",
    schema=QDRANT_DOC_SEARCH_SCHEMA,
    handler=_handle_qdrant_doc_search,
    check_fn=_check_qdrant_available,
    emoji="📄",
)
