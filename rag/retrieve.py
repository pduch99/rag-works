# question -> ranked chunks. vector, bm25 or hybrid (rrf), then an optional reranker. every hit keeps its rank at each step
import json
import math
import os
import re
import threading
import time

from rank_bm25 import BM25Okapi

from .config import CROSS_MODEL, PROVIDERS, ROOT, RRF_K
from .embed import embed
from .index import all_chunks, collection, parse_ix, status
from .llm import chat
from .timing import Timing

_bm25, _cross = {}, {}
STOP = set("a an the of to in on at for is are was were be by and or with from as it its this that what which who where when how do does did i you".split())


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*", text.lower()) if t not in STOP]


def bm25_index(kb, key):
    st = status(kb, key)
    tag = st["built"] if st else None
    hit = _bm25.get((kb, key))
    if hit and hit[0] == tag:
        return hit[1], hit[2]
    chunks = all_chunks(kb, key)
    idx = BM25Okapi([tokens(c["text"]) or ["_"] for c in chunks]) if chunks else None
    _bm25[(kb, key)] = (tag, idx, chunks)
    return idx, chunks


def hit(c, **extra):
    return {"id": c["id"], "text": c["text"], **c["meta"], **extra}


def vector_search(kb, key, question, n, tm=None):
    col = collection(kb, key)
    n = min(n, col.count())
    if n == 0:
        return []
    t = time.perf_counter()
    qv = embed(parse_ix(key)["embedder"], [question], query=True)
    if tm:
        tm.stages.append(("embed question", time.perf_counter() - t, parse_ix(key)["embedder"] == "gemini"))
    r = col.query(query_embeddings=qv, n_results=n, include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": d, **m, "distance": round(dist, 4), "vec_score": round(1 - dist, 4)}
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]


def bm25_search(kb, key, question, n):
    idx, chunks = bm25_index(kb, key)
    q = tokens(question)
    if not idx or not q:
        return []
    scores = idx.get_scores(q)
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:n]
    best = max(float(scores[top[0]]), 1e-9) if top else 1.0
    return [hit(chunks[i], bm25=round(float(scores[i]), 3), bm25_score=round(float(scores[i]) / best, 4)) for i in top if scores[i] > 0]


def rrf(lists):
    """reciprocal rank fusion: score = sum 1/(60 + rank). works on ranks, so scores from different retrievers never get compared"""
    fused = {}
    for lst in lists:
        for r, h in enumerate(lst, start=1):
            f = fused.setdefault(h["id"], {**h, "rrf": 0.0})
            f.update({k: v for k, v in h.items() if k not in f or f[k] is None})
            f["rrf"] += 1 / (RRF_K + r)
    return sorted(fused.values(), key=lambda h: -h["rrf"])


_load_lock = threading.Lock()


def cross_model():
    with _load_lock:
        if "m" not in _cross:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _cross["m"] = TextCrossEncoder(CROSS_MODEL)   # ~80 mb the first time
    return _cross["m"]


RERANK_CHARS = int(os.getenv("RERANK_CHARS", 600))   # cross-encoder cost grows with text length. ~600 chars keeps it fast
RERANK_MAX = int(os.getenv("RERANK_MAX", 10))        # only the top candidates get reranked, the rest keep their order


def cross_rerank(question, hits):
    """measured 1.4 s/question on a laptop cpu when fed 2000-char passages. the start of a chunk carries
    its heading / context line, so the first ~600 chars is enough to judge it"""
    cross_model()
    top, rest = hits[:RERANK_MAX], hits[RERANK_MAX:]
    scores = list(_cross["m"].rerank(question, [h["text"][:RERANK_CHARS] for h in top]))
    for h, s in zip(top, scores):
        h["rerank_score"] = round(1 / (1 + math.exp(-float(s))), 4)
    return sorted(top, key=lambda h: -h["rerank_score"]) + rest


GEMINI_RERANK = """Question: {question}

Below are {n} numbered candidate pieces of evidence. Images for some candidates are attached in order, each labeled with its number.
{cands}

Which candidates help answer the question? Reply with only JSON: {{"order": [numbers, most useful first], "irrelevant": [numbers]}}"""


def llm_rerank(question, hits, provider, max_n=12, tm=None):
    """listwise rerank by the model. it sees the actual crops, not just their descriptions"""
    top, rest = hits[:max_n], hits[max_n:]
    imgs = 0
    content, lines = [], []
    for i, h in enumerate(top, start=1):
        lines.append(f"[{i}] {h['doc']} p{h['page']} {h.get('label', '')}: {h['text'][:400]}")
    content.append(GEMINI_RERANK.format(question=question, n=len(top), cands="\n".join(lines)))
    for i, h in enumerate(top, start=1):
        if h["kind"] == "visual" and h.get("crop") and imgs < PROVIDERS[provider]["max_images"]:
            content += [f"image for [{i}]", ROOT / h["crop"]]
            imgs += 1
    raw, secs = chat(provider, PROVIDERS[provider]["vision" if imgs else "text"], content, max_tokens=400)
    if tm:
        tm.model("gemini rerank", secs)
    try:
        j = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
        order = [int(x) for x in j.get("order", []) if 1 <= int(x) <= len(top)]
        bad = {int(x) for x in j.get("irrelevant", [])}
    except Exception:
        return hits
    seen, out = set(), []
    for n in order + [i for i in range(1, len(top) + 1) if i not in bad] + sorted(bad):
        if n not in seen:
            seen.add(n)
            h = top[n - 1]
            h["rerank_score"] = round(1 - len(out) / max(1, len(top)), 4) if n not in bad else 0.0
            out.append(h)
    return out + rest


def search(kb, key, question, retriever="vector", reranker="none", k=8, pool=20, provider="gemini", tm=None):
    """returns the top k hits. each hit has rank plus vec_rank / bm25_rank / fused_rank so the ui can show who moved what"""
    tm = tm or Timing()
    pool = max(pool, k)
    with tm.stage("vector search"):
        vec = vector_search(kb, key, question, pool, tm) if retriever in ("vector", "hybrid") else []
    with tm.stage("bm25 search"):
        kw = bm25_search(kb, key, question, pool) if retriever in ("bm25", "hybrid") else []
    vrank = {h["id"]: r for r, h in enumerate(vec, start=1)}
    brank = {h["id"]: r for r, h in enumerate(kw, start=1)}
    cands = rrf([vec, kw]) if retriever == "hybrid" else (vec if retriever == "vector" else kw)
    for r, h in enumerate(cands, start=1):
        h["vec_rank"], h["bm25_rank"], h["fused_rank"] = vrank.get(h["id"]), brank.get(h["id"]), r
    with tm.stage(f"rerank ({reranker})"):
        if reranker == "cross" and cands:
            cands = cross_rerank(question, cands)
        elif reranker == "gemini" and cands:
            cands = llm_rerank(question, cands, provider, tm=tm)
    out = cands[:k]
    for r, h in enumerate(out, start=1):
        h["rank"] = r
        h["moved"] = h["fused_rank"] - r if reranker != "none" else 0
        # one 0..1 score to show: whatever ordered it last
        h["score"] = h.get("rerank_score") if reranker != "none" else (h.get("vec_score") if retriever == "vector" else h.get("bm25_score") if retriever == "bm25" else None)
        if h["score"] is None and retriever == "hybrid":
            h["score"] = round(h["rrf"] / (2 / (RRF_K + 1)), 4)
        h.setdefault("distance", None)
    return out