import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import bm25s
import faiss
import numpy as np

from rag.companies import detect_companies

try:
    import Stemmer
    STEMMER = Stemmer.Stemmer("english")
except ImportError:
    STEMMER = None

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ROOT = Path(__file__).resolve().parent.parent
QUERY_CACHE_DIR = ROOT / "data" / "cache"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunks(index_dir):
    return load_jsonl(os.path.join(index_dir, "chunks.jsonl"))


def load_meta(index_dir):
    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


class _Base:
    chunks: list

    def search_ids(self, queries, k, filters=None):
        """filters: None, or one set of company slugs per query (empty set = no filter)."""
        raise NotImplementedError

    def search_batch(self, queries, k):
        return [[(self.chunks[i], s) for i, s in row] for row in self.search_ids(queries, k)]

    def search(self, query, k=5):
        return self.search_batch([query], k)[0]


class DenseRetriever(_Base):
    """Local embedding model (bge-small) for indexes built by rag/index.py. FAISS only."""
    def __init__(self, index_dir, chunks=None, store="faiss"):
        from sentence_transformers import SentenceTransformer
        if store != "faiss":
            raise SystemExit("bge-small indexes are only in FAISS; use --store faiss")
        meta = load_meta(index_dir)
        name = (meta.get("model") or meta.get("model_name")
                or meta.get("embedding_model") or "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(name)
        self.prefix = BGE_QUERY_PREFIX if "bge" in name.lower() else ""
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.chunks = chunks if chunks is not None else load_chunks(index_dir)
        assert self.index.ntotal == len(self.chunks), "index and chunks.jsonl out of sync"
        print(f"[dense] {name}, {len(self.chunks)} chunks")

    def search_ids(self, queries, k, filters=None):
        if filters and any(filters):
            raise SystemExit("company filter needs the Qdrant store (Fin-E5 index, --store qdrant)")
        q = self.model.encode([self.prefix + x for x in queries], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=False)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [[(int(i), float(s)) for i, s in zip(ri, rs) if i != -1]
                for ri, rs in zip(ids, scores)]


class ApiDenseRetriever(_Base):
    """API embedding model (Fin-E5 via AbaciNLP) for indexes built by rag/embed_api.py.

    Queries are embedded through the API with the instruct prefix stored in meta.json and
    cached on disk (data/cache/). Vector search runs in FAISS or in Qdrant (store="qdrant").
    Only Qdrant supports the company filter.
    """

    def __init__(self, index_dir, chunks=None, store="faiss"):
        from dotenv import load_dotenv
        from openai import OpenAI

        meta = load_meta(index_dir)
        self.model_name = meta["model"]
        self.prefix = meta["query_prefix"]
        load_dotenv()
        self.client = OpenAI(api_key=os.environ["ABACI_API_KEY"], base_url=meta["base_url"],
                             max_retries=8, timeout=120)
        self.chunks = chunks if chunks is not None else load_chunks(index_dir)
        self.store = store

        if store == "qdrant":
            from rag.qdrant_store import collection_name, get_client
            self.qdrant = get_client()
            self.collection = collection_name(index_dir)
            n = self.qdrant.count(self.collection, exact=True).count
            assert n == len(self.chunks), f"Qdrant has {n} points, chunks.jsonl has {len(self.chunks)}"
        else:
            self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
            assert self.index.ntotal == len(self.chunks), "index and chunks.jsonl out of sync"

        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.model_name)
        self.cache_path = QUERY_CACHE_DIR / f"queries_{safe}.npz"
        self.cache = self._load_cache()
        print(f"[dense-api] {self.model_name} ({store}), {len(self.chunks)} chunks, "
              f"{len(self.cache)} cached query vectors")

    # ---------- query cache ----------
    def _key(self, query):
        return hashlib.sha1(f"{self.model_name}\x00{self.prefix}{query}".encode("utf-8")).hexdigest()

    def _load_cache(self):
        if not self.cache_path.exists():
            return {}
        d = np.load(self.cache_path)
        return {str(k): v for k, v in zip(d["keys"], d["vecs"])}

    def _save_cache(self):
        QUERY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))
        os.replace(tmp, self.cache_path)

    # ---------- embedding ----------
    def encode(self, queries):
        from rag.embed_api import create_with_retry

        keys = [self._key(q) for q in queries]
        missing = {k: q for k, q in zip(keys, queries) if k not in self.cache}
        if missing:
            items = list(missing.items())
            for s in range(0, len(items), 32):
                batch = items[s:s + 32]
                resp = create_with_retry(self.client, self.model_name,
                                         [self.prefix + q for _, q in batch])
                data = sorted(resp.data, key=lambda d: d.index)
                for (k, _), d in zip(batch, data):
                    v = np.asarray(d.embedding, dtype=np.float32)
                    self.cache[k] = v / max(float(np.linalg.norm(v)), 1e-12)
            self._save_cache()
            print(f"[dense-api] embedded {len(missing)} new queries via API (cached for next time)")
        return np.stack([self.cache[k] for k in keys]).astype(np.float32)

    # ---------- search ----------
    def search_ids(self, queries, k, filters=None):
        vecs = self.encode(queries)
        if self.store == "qdrant":
            from qdrant_client import models
            from rag.qdrant_store import company_filter, is_server
            # exact=True on a server = brute force, same results as FAISS (local mode is always exact)
            params = models.SearchParams(exact=True) if is_server() else None
            out = []
            for i, v in enumerate(vecs):
                res = self.qdrant.query_points(
                    self.collection, query=v.tolist(), limit=k,
                    query_filter=company_filter(filters[i] if filters else None),
                    search_params=params, with_payload=False)
                out.append([(int(p.id), float(p.score)) for p in res.points])
            return out
        if filters and any(filters):
            raise SystemExit("company filter needs --store qdrant")
        scores, ids = self.index.search(vecs, k)
        return [[(int(i), float(s)) for i, s in zip(ri, rs) if i != -1]
                for ri, rs in zip(ids, scores)]


def make_dense(index_dir, chunks=None, store="faiss"):
    """Pick the dense retriever from meta.json: API-embedded indexes vs local bge indexes."""
    cls = ApiDenseRetriever if load_meta(index_dir).get("embedder") == "api" else DenseRetriever
    return cls(index_dir, chunks, store=store)


class BM25Retriever(_Base):
    def __init__(self, index_dir, chunks=None):
        self.chunks = chunks if chunks is not None else load_chunks(index_dir)
        tokens = bm25s.tokenize([c["text"] for c in self.chunks], stopwords="en",
                                stemmer=STEMMER, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)
        self.companies = np.array([c["company"] for c in self.chunks])
        print(f"[bm25] indexed {len(self.chunks)} chunks (stemmer={'on' if STEMMER else 'off'})")

    def search_ids(self, queries, k, filters=None):
        q = bm25s.tokenize(queries, stopwords="en", stemmer=STEMMER,
                           return_ids=False, show_progress=False)
        if not (filters and any(filters)):  # unfiltered: unchanged Day 2 behaviour
            ids, scores = self.bm25.retrieve(q, k=min(k, len(self.chunks)), show_progress=False)
            return [[(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(ids, scores)]
        out = []
        for tokens, comp in zip(q, filters):
            scores = self.bm25.get_scores(tokens)
            if comp:
                scores = np.where(np.isin(self.companies, list(comp)), scores, -np.inf)
            top = np.argsort(-scores)[:k]
            out.append([(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])])
        return out


class HybridRetriever(_Base):
    def __init__(self, index_dir, rrf_k=60, fetch_k=50, store="faiss", company_filter=False):
        self.chunks = load_chunks(index_dir)
        self.dense = make_dense(index_dir, self.chunks, store=store)
        self.bm25 = BM25Retriever(index_dir, self.chunks)
        self.rrf_k, self.fetch_k = rrf_k, fetch_k
        self.company_filter = company_filter
        self.last_filters = []

    def search_ids(self, queries, k, filters=None):
        if filters is None and self.company_filter:
            filters = [detect_companies(q) for q in queries]
        self.last_filters = filters or [set() for _ in queries]
        dense = self.dense.search_ids(queries, self.fetch_k, filters)
        sparse = self.bm25.search_ids(queries, self.fetch_k, filters)
        out = []
        for d, b in zip(dense, sparse):
            fused = {}
            for ranked in (d, b):
                for rank, (i, _) in enumerate(ranked, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
            out.append(sorted(fused.items(), key=lambda x: -x[1])[:k])
        return out


def make_retriever(mode, index_dir, store="faiss", company_filter=False):
    if company_filter and mode != "hybrid":
        raise SystemExit("--company-filter is implemented for --mode hybrid")
    if mode == "dense":
        return make_dense(index_dir, store=store)
    if mode == "bm25":
        return BM25Retriever(index_dir)
    return HybridRetriever(index_dir, store=store, company_filter=company_filter)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default="data/index/docling")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--store", choices=["faiss", "qdrant"], default="faiss")
    ap.add_argument("--company-filter", action="store_true")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    r = make_retriever(a.mode, a.index, store=a.store, company_filter=a.company_filter)
    results = r.search(a.query, a.k)
    if a.company_filter:
        print(f"\ncompany filter: {sorted(r.last_filters[0]) or 'none detected'}")
    for rank, (c, s) in enumerate(results, 1):
        print(f"\n#{rank}  {c['company']} p{c['page']}  [{c.get('type')}]  score={s:.4f}")
        print(c["text"][:300])