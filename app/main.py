"""Annualyst RAG — HTTP API.

Endpoints:
    GET  /health     -> is everything loaded? (chunks, Qdrant, model)
    GET  /companies  -> companies the user can pick
    POST /ask        -> {"question": "...", "company": null}  ->  answer + citations + timings

Run (from the repo root, Qdrant running):
    uv run uvicorn app.main:app --port 8000
Then open http://localhost:8000/docs to try it in the browser.
"""
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.answer import COMPANY_NAMES, DEFAULT_INDEX, DEFAULT_MODEL, Answerer

INDEX_DIR = os.getenv("ANNUALYST_INDEX", DEFAULT_INDEX)
STORE = os.getenv("ANNUALYST_STORE", "qdrant")

state = {}
lock = threading.Lock()  # one question at a time: the query cache file and clients are shared


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup (BM25 over 20k chunks, Qdrant client) — not on every request.
    state["answerer"] = Answerer(INDEX_DIR, store=STORE, company_filter=True, on_no_company="ask")
    yield
    state.clear()


app = FastAPI(title="Annualyst RAG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["What was HDFC Bank's standalone net profit in FY26?"])
    company: Optional[str] = Field(None, description="company slug from /companies; overrides detection")
    search_all: bool = Field(False, description="if no company is named, search all reports instead of asking")


class Source(BaseModel):
    company: str
    company_name: str
    page: int
    chunk_id: str
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    reason: Optional[str]
    companies: list[str]
    sources: list[Source]
    timings: dict[str, float]
    tokens: dict[str, int]
    cost_usd: float


@app.get("/health")
def health():
    a = state.get("answerer")
    if a is None:
        raise HTTPException(503, "still loading")
    return {"status": "ok", "chunks": len(a.retriever.chunks), "index": INDEX_DIR,
            "store": STORE, "model": a.model}


@app.get("/companies")
def companies():
    return [{"slug": slug, "name": name} for slug, name in COMPANY_NAMES.items()]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    a = state.get("answerer")
    if a is None:
        raise HTTPException(503, "still loading")
    if req.company and req.company not in COMPANY_NAMES:
        raise HTTPException(422, f"unknown company '{req.company}', see /companies")

    with lock:
        prev = a.on_no_company
        a.on_no_company = "search_all" if req.search_all else "ask"
        try:
            r = a.answer(req.question.strip(), company=req.company)
        finally:
            a.on_no_company = prev

    by_id = {h["chunk_id"]: h for h in r["hits"]}
    sources = [Source(company=c["company"], company_name=COMPANY_NAMES.get(c["company"], c["company"]),
                      page=int(c["page"]), chunk_id=c["chunk_id"],
                      snippet=" ".join(by_id[c["chunk_id"]]["text"].split())[:400])
               for c in r["citations"]]
    return AskResponse(question=r["question"], answer=r["answer"], found=r["found"],
                       reason=r["reason"], companies=r["companies"], sources=sources,
                       timings=r["timings"], tokens=r["usage"], cost_usd=r["cost_usd"])