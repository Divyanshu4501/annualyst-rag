"""Load an existing index folder (embeddings.npy + chunks.jsonl) into a Qdrant collection.

No embedding API calls: it reuses the vectors already on disk.

Where Qdrant runs:
    - QDRANT_URL set (e.g. http://localhost:6333)  -> Qdrant server (Docker)
    - QDRANT_URL not set                           -> local mode, stored in data/qdrant/

Usage:
    uv run python -m rag.qdrant_store --index data/index/docling_fine5
"""
import argparse
import atexit
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATH = ROOT / "data" / "qdrant"
PAYLOAD_FIELDS = ("chunk_id", "company", "page", "type", "section", "source", "text")


_CLIENT = None


def is_server():
    load_dotenv()
    return bool(os.getenv("QDRANT_URL"))


def get_client():
    """One shared client per process (local mode allows only one client per folder)."""
    global _CLIENT
    if _CLIENT is None:
        if is_server():
            _CLIENT = QdrantClient(url=os.environ["QDRANT_URL"],api_key=os.getenv("QDRANT_API_KEY"), timeout=120)
        else:
            _CLIENT = QdrantClient(path=str(LOCAL_PATH))
        atexit.register(_CLIENT.close)  # close cleanly before Python shuts down
    return _CLIENT


def collection_name(index_dir):
    return os.path.basename(os.path.normpath(index_dir))


def company_filter(companies):
    """Qdrant filter for a set of company slugs (None = no filter)."""
    if not companies:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="company", match=models.MatchAny(any=sorted(companies)))])


def build(index_dir, name=None, batch=128):
    name = name or collection_name(index_dir)
    emb = np.load(os.path.join(index_dir, "embeddings.npy"), mmap_mode="r")
    with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    n, dim = emb.shape
    if n != len(chunks):
        raise SystemExit(f"embeddings ({n}) and chunks ({len(chunks)}) out of sync")

    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(
        size=dim, distance=models.Distance.COSINE))
    if is_server():  # speeds up company filtering on a server; local mode has no indexes
        client.create_payload_index(name, field_name="company",
                                    field_schema=models.PayloadSchemaType.KEYWORD)

    t0 = time.time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        client.upsert(name, wait=True, points=models.Batch(
            ids=list(range(s, e)),
            vectors=np.asarray(emb[s:e], dtype=np.float32).tolist(),
            payloads=[{k: c.get(k) for k in PAYLOAD_FIELDS} for c in chunks[s:e]]))
        print(f"\r  uploaded {e:,}/{n:,}", end="", flush=True)
    count = client.count(name, exact=True).count
    print(f"\n[qdrant] collection '{name}': {count:,} points, dim {dim}, {time.time() - t0:.0f}s")
    if count != n:
        raise SystemExit(f"expected {n} points, got {count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index folder with embeddings.npy + chunks.jsonl")
    ap.add_argument("--collection", default=None, help="default: index folder name")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    build(a.index, a.collection, a.batch)