"""Answer a question from the annual reports with page citations.

Pipeline (best config from the eval):
    company detection -> Fin-E5 query embedding (API + cache) -> Qdrant (filtered) + BM25 (filtered)
    -> RRF fusion -> top k chunks -> gpt-4o-mini (strict JSON) -> citation validation

Usage:
    uv run python -m rag.answer "What was HDFC Bank's net profit in FY26?"
    uv run python -m rag.answer "What was the capital adequacy ratio?" --company hdfcbank
    uv run python -m rag.answer "..." --show-context
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.companies import COMPANY_ALIASES, detect_companies
from rag.hybrid import make_retriever

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_INDEX = "data/index/docling_fine5"
# USD per 1M tokens, for the cost estimate only (check current OpenAI pricing and override in .env)
PRICE_IN = float(os.getenv("OPENAI_PRICE_IN_PER_M", "0.15"))
PRICE_OUT = float(os.getenv("OPENAI_PRICE_OUT_PER_M", "0.60"))

COMPANY_NAMES = {
    "asianpaints": "Asian Paints", "eternal": "Eternal (Zomato)", "hdfcbank": "HDFC Bank",
    "icicibank": "ICICI Bank", "infosys": "Infosys", "itc": "ITC", "nykaa": "Nykaa (FSN E-Commerce)",
    "titan": "Titan",
}

SYSTEM_PROMPT = """You answer questions about Indian company annual reports (FY2025-26).

Rules:
1. Use ONLY the context chunks provided. Never use outside knowledge.
2. Every factual claim must be supported by at least one chunk; list the IDs of the chunks you used in "citations".
3. If the context does not contain the answer, set "found" to false, set "answer" to a one-line statement that the reports provided do not contain this information, and return an empty "citations" list. Do not guess.
4. Numbers: copy every figure EXACTLY as printed in the context (same digits, commas and decimal point) and say which year it belongs to. Never move a decimal point or rescale on your own.
5. Note: due to PDF font extraction, the rupee symbol may appear as ` or C or I or H or J (e.g. `74,671.3 crore, I3,110 crore, "(C in '000)"). Treat these as ₹.
6. Distinguish the company from its subsidiaries (e.g. HDFC Bank vs HDFC Securities). Only answer about the entity asked for.
7. Tables are given in markdown. Read the column headers carefully (current year vs previous year, standalone vs consolidated).
8. Units: a block header may say "units on this page: ...". That unit applies to every figure from that page. Answer as: the figure exactly as printed, its unit, then the ₹ crore value, e.g. "746,712,933 (₹ in '000), i.e. ₹74,671.29 crore". Convert using exactly: 1 crore = 100 lakh = 10,000 thousand = 10 million. If no unit is given for a figure, give it as printed and say the unit is not stated in the retrieved text. Never write "crore" next to a number that is not in crore."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "found": {"type": "boolean"},
            },
            "required": ["answer", "citations", "found"],
            "additionalProperties": False,
        },
    },
}

# Unit declarations printed once per page, e.g. "(C in '000)" (the rupee sign often extracts as
# ` C I H J), "(₹ in lakh)", "(₹ crore)". Found on any chunk of a page -> applied to the whole page.
UNIT_RE = re.compile(
    r"\(\s*(?:₹|`|C|I|H|J|Rs\.?|INR)?\s*(?:in\s+)?(['‘’]\s?000|thousands?|lakhs?|millions?|crores?)\s*\)",
    re.IGNORECASE)
UNIT_NOTES = {
    "000": "₹ in thousands ('000) - divide by 10,000 to get ₹ crore",
    "thousand": "₹ in thousands - divide by 10,000 to get ₹ crore",
    "lakh": "₹ in lakh - divide by 100 to get ₹ crore",
    "million": "₹ in million - divide by 10 to get ₹ crore",
    "crore": "₹ in crore",
}


def page_unit(texts):
    """Return a unit note if any chunk of the page declares a unit, else None."""
    for t in texts:
        m = UNIT_RE.search(t)
        if m:
            u = m.group(1).lower()
            for key, note in UNIT_NOTES.items():
                if key in u:
                    return note
    return None


NO_COMPANY_MSG = ("Which company do you mean? The question doesn't name one. "
                  "Available: " + ", ".join(COMPANY_NAMES.values()) + ".")
NOT_VERIFIED_MSG = "I couldn't verify an answer to this in the provided reports."


def build_context(hits):
    blocks = []
    for h in hits:
        name = COMPANY_NAMES.get(h["company"], h["company"])
        section = f", section: {h['section']}" if h.get("section") else ""
        unit = f", units on this page: {h['unit']}" if h.get("unit") else ""
        blocks.append(f"[{h['chunk_id']}] ({name}, page {h['page']}{section}{unit})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


class Answerer:
    def __init__(self, index_dir=DEFAULT_INDEX, store="qdrant", company_filter=True,
                 on_no_company="ask", model=DEFAULT_MODEL, k=10, pages=4, max_chars=12000):
        """k: chunks retrieved; pages: best distinct pages expanded to ALL their chunks;
        max_chars: cap on total context sent to the LLM (~4 chars per token).
        on_no_company: "ask" = clarifying question when no company is named,
        "search_all" = search every company's report."""
        self.retriever = make_retriever("hybrid", index_dir, store=store,
                                        company_filter=company_filter)
        self.company_filter = company_filter
        self.on_no_company = on_no_company
        self.client = OpenAI()
        self.model = model
        self.k, self.pages, self.max_chars = k, pages, max_chars
        # (company, page) -> chunk indices in document order, for small-to-big page expansion
        self.page_chunks = defaultdict(list)
        for i, c in enumerate(self.retriever.chunks):
            self.page_chunks[(c["company"], c["page"])].append(i)

    def expand_to_pages(self, ids):
        """Small-to-big: retrieve small chunks, but give the LLM whole pages.
        A heading chunk ("STANDALONE PROFIT AND LOSS ACCOUNT") pulls in the table on the same page."""
        chunks = self.retriever.chunks
        retrieved = {i for i, _ in ids}
        page_order, page_score = [], {}
        for i, s in ids:
            key = (chunks[i]["company"], chunks[i]["page"])
            if key not in page_score:
                page_order.append(key)
                page_score[key] = s
        hits, used = [], 0
        for key in page_order[:self.pages]:
            unit = page_unit(chunks[j]["text"] for j in self.page_chunks[key])
            for i in self.page_chunks[key]:
                text = chunks[i]["text"]
                if used + len(text) > self.max_chars and hits:
                    break
                used += len(text)
                c = chunks[i]
                hits.append({"score": page_score[key], "chunk_id": c["chunk_id"],
                             "company": c["company"], "page": c["page"], "type": c.get("type"),
                             "section": c.get("section"), "text": text, "unit": unit,
                             "retrieved": i in retrieved})
        return hits

    def _empty(self, question, answer, companies, timings, reason):
        return {"question": question, "answer": answer, "found": False, "reason": reason,
                "companies": sorted(companies), "citations": [], "dropped_citations": [],
                "hits": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "cost_usd": 0.0, "timings": timings}

    def answer(self, question, company=None):
        t0 = time.perf_counter()
        timings = {}

        # 1. which company? explicit choice (e.g. a UI dropdown) beats detection
        if company:
            if company not in COMPANY_ALIASES:
                raise ValueError(f"unknown company '{company}', use one of {list(COMPANY_ALIASES)}")
            companies = {company}
        else:
            companies = detect_companies(question) if self.company_filter else set()
        if self.company_filter and not companies and self.on_no_company == "ask":
            timings["total_s"] = round(time.perf_counter() - t0, 3)
            return self._empty(question, NO_COMPANY_MSG, companies, timings, "no_company")

        # 2. embed the question (API call, or instant if cached) - timed on its own
        dense = getattr(self.retriever, "dense", None)
        if hasattr(dense, "encode"):
            t = time.perf_counter()
            dense.encode([question])
            timings["embed_s"] = round(time.perf_counter() - t, 3)

        # 3. retrieve: Qdrant + BM25 (both filtered) + RRF, then expand the best pages
        t = time.perf_counter()
        ids = self.retriever.search_ids([question], self.k, filters=[companies])[0]
        hits = self.expand_to_pages(ids)
        timings["retrieve_s"] = round(time.perf_counter() - t, 3)
        if not hits:
            timings["total_s"] = round(time.perf_counter() - t0, 3)
            return self._empty(question, NOT_VERIFIED_MSG, companies, timings, "no_hits")

        # 4. generate
        t = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format=RESPONSE_SCHEMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n\n{build_context(hits)}\n\nQuestion: {question}"},
            ],
        )
        timings["llm_s"] = round(time.perf_counter() - t, 3)
        out = json.loads(resp.choices[0].message.content)

        # 5. validate citations: only chunks we actually gave the model count
        by_id = {h["chunk_id"]: h for h in hits}
        valid = [cid for cid in dict.fromkeys(out["citations"]) if cid in by_id]
        dropped = [cid for cid in out["citations"] if cid not in by_id]
        found, answer, reason = out["found"], out["answer"], None
        if found and not valid:  # claims an answer but can't point to evidence -> refuse
            found, answer, reason = False, NOT_VERIFIED_MSG, "no_valid_citations"
        elif not found:
            reason = "not_in_context"

        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens}
        timings["total_s"] = round(time.perf_counter() - t0, 3)
        return {
            "question": question,
            "answer": answer,
            "found": found,
            "reason": reason,
            "companies": sorted(companies),
            "citations": [{"chunk_id": cid, "company": by_id[cid]["company"],
                           "page": by_id[cid]["page"]} for cid in valid] if found else [],
            "dropped_citations": dropped,
            "hits": hits,
            "usage": usage,
            "cost_usd": round((usage["prompt_tokens"] * PRICE_IN
                               + usage["completion_tokens"] * PRICE_OUT) / 1e6, 6),
            "timings": timings,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--company", default=None, help="force a company slug, e.g. hdfcbank")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--store", choices=["faiss", "qdrant"], default="qdrant")
    ap.add_argument("--no-company-filter", action="store_true")
    ap.add_argument("--search-all", action="store_true",
                    help="if no company is named, search all reports instead of asking")
    ap.add_argument("--k", type=int, default=10, help="chunks to retrieve")
    ap.add_argument("--pages", type=int, default=4, help="best pages expanded into the context")
    ap.add_argument("--max-chars", type=int, default=12000, help="context size cap")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    a = ap.parse_args()

    answerer = Answerer(a.index, store=a.store, company_filter=not a.no_company_filter,
                        on_no_company="search_all" if a.search_all else "ask",
                        model=a.model, k=a.k, pages=a.pages, max_chars=a.max_chars)
    r = answerer.answer(a.question, company=a.company)

    if a.json:
        print(json.dumps({k: v for k, v in r.items() if k != "hits"}, indent=2, ensure_ascii=False))
        return
    if a.show_context:
        print(f"\n=== context sent to the LLM ({len(r['hits'])} chunks, "
              f"{sum(len(h['text']) for h in r['hits']):,} chars; * = directly retrieved) ===")
        for i, h in enumerate(r["hits"], 1):
            star = "*" if h.get("retrieved") else " "
            print(f"{star}{i:>2} {h['company']} p{h['page']} [{h['type']}] {h['chunk_id']}  "
                  f"{' '.join(h['text'].split())[:150]}...")

    print(f"\nQ: {r['question']}")
    print(f"company filter: {r['companies'] or 'none'}")
    print(f"\nA: {r['answer']}")
    print(f"\nfound: {r['found']}" + (f"  (reason: {r['reason']})" if r["reason"] else ""))
    if r["citations"]:
        print("\nSources:")
        for c in r["citations"]:
            print(f"  - {COMPANY_NAMES.get(c['company'], c['company'])} p.{c['page']}  [{c['chunk_id']}]")
    if r["dropped_citations"]:
        print(f"\n(warning: model cited unknown chunks, dropped: {r['dropped_citations']})")
    u, t = r["usage"], r["timings"]
    print(f"\n[tokens: {u['prompt_tokens']} in / {u['completion_tokens']} out | "
          f"cost ≈ ${r['cost_usd']:.5f} | model: {a.model}]")
    print("[time: " + " | ".join(f"{k} {v}s" for k, v in t.items()) + "]")


if __name__ == "__main__":
    main()