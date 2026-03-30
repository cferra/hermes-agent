#!/usr/bin/env python3
"""
qdrant_ingest.py — Document ingestion into Qdrant hermes_docs collection.

Usage:
    python3 qdrant_ingest.py <file_or_directory> [options]

Options:
    --chunk-size   Characters per chunk (default: 1500, ~375 tokens)
    --overlap      Overlap between chunks in chars (default: 200)
    --clear        Delete existing chunks from this source before re-ingesting
    --collection   Target collection (default: hermes_docs)
    --extensions   Comma-separated list of file extensions to include
                   (default: md,txt,py,yaml,yml,json,rst,sh)

Examples:
    python3 qdrant_ingest.py ~/Documents/FerranteAI-Cluster-Setup.md
    python3 qdrant_ingest.py ~/Documents/ --extensions md,txt
    python3 qdrant_ingest.py ~/code/myproject/ --extensions py --clear
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add hermes-agent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from qdrant_memory import get_embedding, QDRANT_URL, BGE_EMBED_URL, VECTOR_SIZE
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    PointIdsList,
)

DEFAULT_COLLECTION = "hermes_docs"
DEFAULT_CHUNK_SIZE = 1500   # ~375 tokens — fits comfortably in most context windows
DEFAULT_OVERLAP = 200       # ~50 token overlap to preserve continuity


def ensure_docs_collection(client: QdrantClient, collection: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, breaking on word boundaries.
    Tries to keep markdown section headers with their content.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Try to break at a paragraph boundary first
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Fall back to word boundary
                word_break = text.rfind(" ", start, end)
                if word_break > start + chunk_size // 2:
                    end = word_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end - overlap)

    return chunks


def clear_source(client: QdrantClient, collection: str, source_path: str) -> int:
    """Delete all chunks from a specific source file."""
    try:
        # Scroll to find all points with this source
        deleted = 0
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchAny(any=[source_path]))]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
            )
            if not points:
                break
            ids = [p.id for p in points]
            client.delete(collection_name=collection, points_selector=PointIdsList(points=ids))
            deleted += len(ids)
            if next_offset is None:
                break
            offset = next_offset
        return deleted
    except Exception:
        return 0


def ingest_file(
    client: QdrantClient,
    filepath: Path,
    collection: str,
    chunk_size: int,
    overlap: int,
    clear: bool,
) -> int:
    """Ingest a single file. Returns number of chunks upserted."""
    source_key = str(filepath.resolve())

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        print(f"  ✗ Could not read {filepath.name}: {e}")
        return 0

    if not text:
        print(f"  ✗ Skipping empty file: {filepath.name}")
        return 0

    if clear:
        removed = clear_source(client, collection, source_key)
        if removed:
            print(f"  🗑  Cleared {removed} existing chunks from {filepath.name}")

    chunks = chunk_text(text, chunk_size, overlap)
    total = len(chunks)

    points = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        point_id = uuid.uuid4().hex
        payload = {
            "text": chunk,
            "source": source_key,
            "filename": filepath.name,
            "chunk_index": i,
            "total_chunks": total,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        # Batch upsert every 20 chunks
        if len(points) >= 20:
            client.upsert(collection_name=collection, points=points)
            print(f"  ↑ {filepath.name}: {i+1}/{total} chunks", end="\r")
            points = []

    if points:
        client.upsert(collection_name=collection, points=points)

    print(f"  ✓ {filepath.name}: {total} chunks ingested                    ")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant hermes_docs")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--clear", action="store_true", help="Remove existing chunks before re-ingesting")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--extensions", default="md,txt,py,yaml,yml,json,rst,sh,conf,toml")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        print(f"Error: path does not exist: {target}")
        sys.exit(1)

    extensions = {f".{e.lstrip('.')}" for e in args.extensions.split(",")}

    # Collect files
    if target.is_file():
        files = [target]
    else:
        files = sorted([
            f for f in target.rglob("*")
            if f.is_file() and f.suffix in extensions
            and not any(part.startswith(".") for part in f.parts)
        ])

    if not files:
        print(f"No files found matching extensions: {extensions}")
        sys.exit(0)

    print(f"Connecting to Qdrant at {QDRANT_URL}")
    print(f"Embeddings via bge-m3 at {BGE_EMBED_URL}")
    print(f"Collection: {args.collection}")
    print(f"Chunk size: {args.chunk_size} chars, overlap: {args.overlap} chars")
    print(f"Files to ingest: {len(files)}\n")

    client = QdrantClient(url=QDRANT_URL, timeout=15)
    ensure_docs_collection(client, args.collection)

    total_chunks = 0
    for filepath in files:
        n = ingest_file(client, filepath, args.collection, args.chunk_size, args.overlap, args.clear)
        total_chunks += n

    info = client.get_collection(args.collection)
    print(f"\nDone. {total_chunks} chunks ingested.")
    print(f"Collection '{args.collection}' now has {info.points_count} total points.")


if __name__ == "__main__":
    main()
