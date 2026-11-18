# command line version of the app. handy for long eval runs
# run from the rag-works folder:
#   python cli.py kbs                                              list kbs, built indexes
#   python cli.py ping                                             one tiny call to each model, shows the api's real error if any
#   python cli.py index metrostate layout --embedder bge --style structured --contextual
#   python cli.py ask metrostate layout "where is the library?" --retriever hybrid --reranker cross
#   python cli.py eval metrostate layout --embedder bge --retriever hybrid --reranker cross --split dev
#   python cli.py ladder testbed fp7 --split dev                   builds what it needs + runs the whole fp7 experiment
#   python cli.py ladder testbed fp8 --index layout.bge.structured.ctx --split dev
#   python cli.py bench metrostate layout --contextual --retriever hybrid --reranker cross   times our code only, no api calls
#   python cli.py ladder testbed fp9 --index layout.bge.structured.ctx --retriever hybrid --reranker cross --split dev
import argparse

from rag import evals, index, kb
from rag.answer import ask
from rag.config import CHECKS, CHUNKERS, DEFAULT_EMBEDDER, DEFAULT_PROVIDER, DESCRIBE_STYLES, EMBEDDERS, PROVIDERS, RERANKERS, RETRIEVERS, ix_key, parse_ix

# the experiments. each step changes one thing from the step before
LADDERS = {
    "fp7": [  # index changes, plain vector retrieval
        ("fp6 baseline", ("region", "minilm", "prose", False), {}),
        ("+ bge embedder", ("region", "bge", "prose", False), {}),
        ("+ structured descriptions", ("region", "bge", "structured", False), {}),
        ("+ structure text chunks", ("structure", "bge", "structured", False), {}),
        ("+ layout figure chunks", ("layout", "bge", "structured", False), {}),
        ("+ contextual chunks", ("layout", "bge", "structured", True), {}),
    ],
    "fp8": [  # retrieval changes on one fixed index
        ("vector", None, {"retriever": "vector"}),
        ("bm25", None, {"retriever": "bm25"}),
        ("hybrid", None, {"retriever": "hybrid"}),
        ("vector + cross rerank", None, {"retriever": "vector", "reranker": "cross"}),
        ("hybrid + cross rerank", None, {"retriever": "hybrid", "reranker": "cross"}),
        ("hybrid + gemini rerank", None, {"retriever": "hybrid", "reranker": "gemini"}),
    ],
    "fp9": [  # answer-side changes on the best fp8 setup (pass --retriever / --reranker)
        ("no labels, no check", None, {"labels": False, "check": "none"}),
        ("+ label positions", None, {"labels": True, "check": "none"}),
        ("+ grade evidence", None, {"labels": True, "check": "grade"}),
        ("+ verify answer", None, {"labels": True, "check": "verify"}),
        ("full (grade + verify)", None, {"labels": True, "check": "full"}),
        ("auto (verify when unsure)", None, {"labels": True, "check": "auto"}),
    ],
}

p = argparse.ArgumentParser(description="rag works")
sub = p.add_subparsers(dest="cmd", required=True)
sub.add_parser("kbs")
sp = sub.add_parser("ping")
sp.add_argument("--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS))
for name in ("index", "ask", "eval", "ladder", "bench"):
    s = sub.add_parser(name)
    s.add_argument("kb")
    if name == "ladder":
        s.add_argument("ladder", choices=list(LADDERS))
        s.add_argument("--index", help="index key for the fp8 ladder, e.g. layout.bge.structured.ctx")
    else:
        s.add_argument("chunker", choices=list(CHUNKERS))
    if name == "ask":
        s.add_argument("question")
    s.add_argument("--embedder", default=DEFAULT_EMBEDDER, choices=list(EMBEDDERS))
    s.add_argument("--style", default="structured", choices=list(DESCRIBE_STYLES))
    s.add_argument("--contextual", action="store_true")
    s.add_argument("--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS))
    s.add_argument("--retriever", default="vector", choices=list(RETRIEVERS))
    s.add_argument("--reranker", default="none", choices=list(RERANKERS))
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--pool", type=int, default=20)
    s.add_argument("--max-images", type=int, default=2)
    s.add_argument("--max-text", type=int, default=2)
    s.add_argument("--with-page", action="store_true", help="also send the whole page for orientation")
    s.add_argument("--check", default="none", choices=list(CHECKS))
    s.add_argument("--no-labels", action="store_true", help="don't add text-layer label positions to visual evidence")
    s.add_argument("--spatial", action="store_true", help="add directions between the places the question names, computed from label positions (fp10)")
    s.add_argument("--workers", type=int, default=None, help="questions answered at once during evals")
    s.add_argument("--no-judge", action="store_true")
    s.add_argument("--split", default="all", choices=["all", "dev", "test"])
    s.add_argument("--only", nargs="*", help="gold ids to run, like map-03 txt-01")
a = p.parse_args()


def ensure(kbn, key, provider):
    st = index.status(kbn, key)
    if not st or st["stale"]:
        c = parse_ix(key)
        print(f"building {key}")
        index.build(kbn, c["chunker"], provider, c["embedder"], c["style"], c["contextual"])


def report(out):
    for i in out["items"]:
        print(f"  {i['id']:8} retrieval {'PASS' if i['retrieval_pass'] else 'FAIL'} | r@5 {i.get('recall@5')} | answer {i.get('grade') or '?':8} | {i['diagnosis']}")
    s = out["summary"]
    print(f"  -> retrieval {s['retrieval_pass']} | recall@5 {s['recall@5']} | mrr {s['mrr']} | answer {s['answer_score']} | grounded {s['grounded']}"
          f" | false abstain {s['false_abstain']} | hallucinated {s['hallucinated']} | {s['llm_calls']} calls/q")
    print(f"  -> time per question: {s['avg_seconds']}s total, {s['program_ms']} ms our code, rest is the model api. run took {out['wall_seconds']}s")
    print(f"  saved runs/{out['kb']}/{out['file']}")


if a.cmd == "ping":
    from rag.llm import ping
    for m in sorted({PROVIDERS[a.provider]["text"], PROVIDERS[a.provider]["vision"]}):
        ok, msg = ping(a.provider, m)
        print(("ok   " if ok else "FAIL ") + msg)
    raise SystemExit
if a.cmd == "kbs":
    for name in kb.list_kbs():
        print(f"{name}: {len(kb.docs(name))} docs, {len(evals.gold(name))} gold questions")
        for key in index.built(name):
            st = index.status(name, key)
            print(f"  {key:32} {st['chunks']} chunks, built {st['built']}{' (STALE, rebuild)' if st['stale'] else ''}")
    raise SystemExit
q = {"retriever": a.retriever, "reranker": a.reranker, "k": a.k, "pool": a.pool, "max_images": a.max_images, "max_text": a.max_text,
     "with_page": a.with_page, "check": a.check, "labels": not a.no_labels, "spatial": a.spatial}
if a.cmd == "ladder":
    for step, cfg, qx in LADDERS[a.ladder]:
        key = ix_key(*cfg) if cfg else (a.index or ix_key("layout", a.embedder, a.style, a.contextual))
        print(f"\n=== {step}: {key} {qx}")
        ensure(a.kb, key, a.provider)
        report(evals.run(a.kb, key, a.provider, {**q, **qx}, not a.no_judge, a.only, a.split, note=f"{a.ladder}: {step}",
                         progress=lambda i, n, qq: print(f"[{i}/{n}] {qq}"), workers=a.workers))
    raise SystemExit
key = ix_key(a.chunker, a.embedder, a.style, a.contextual)
if a.cmd == "index":
    index.build(a.kb, a.chunker, a.provider, a.embedder, a.style, a.contextual)
elif a.cmd == "ask":
    ensure(a.kb, key, a.provider)
    r = ask(a.kb, key, a.question, a.provider, q)
    print("\nretrieved:")
    for h in r["hits"]:
        mv = f" moved {h['moved']:+d}" if h.get("moved") else ""
        print(f"  {'>' if h['sent'] else ' '} {h['rank']}. {h['doc']} p{h['page']} {h['kind']} {h.get('label', '')} score {h['score']}"
              f" (vec {h.get('vec_rank')}, bm25 {h.get('bm25_rank')}){mv}")
    print("  (> = sent to the model)")
    print(f"\nanswer: {r['answer']}\nwhy: {r['why']}")
    c = r["check"]
    if c["mode"] != "none":
        print(f"check {c['mode']}: confidence {c['confidence']} | abstained {c['abstained']} | retried {c['retried']} | corrected {c['corrected']}")
        for cl in c["claims"] or []:
            print(f"  {'ok ' if cl['supported'] else 'BAD'} {cl['claim']} {cl['evidence']}")
    print("citations: " + (", ".join(f"[{c['n']}] {c['label']}" for c in r["citations"]) or "none"))
    t = r["timing"]
    print(f"{r['model']} | {r['llm_calls']} model calls | {t['total']}s total = {t['program']*1000:.0f} ms our code + {t['model']}s model api")
    print("  " + " · ".join(f"{s['name']} {s['ms']:.0f}ms" + (" (api)" if s["model"] else "") for s in t["stages"]))
elif a.cmd == "bench":
    # every gold question through retrieval + labels + evidence selection, 3 rounds. no model calls, so this is purely our code
    import statistics
    import time
    from rag.answer import labels_for, select
    from rag.retrieve import search
    from rag.timing import Timing
    ensure(a.kb, key, a.provider)
    qs = [g["question"] for g in evals.gold(a.kb, a.split)] or ["test question"]
    search(a.kb, key, qs[0], a.retriever, "none" if a.reranker == "gemini" else a.reranker, a.k, a.pool)   # warm models
    totals, per = [], {}
    for _ in range(3):
        for qq in qs:
            tm = Timing()
            hits = search(a.kb, key, qq, a.retriever, "none" if a.reranker == "gemini" else a.reranker, a.k, a.pool, a.provider, tm)
            with tm.stage("labels + select"):
                for h in hits:
                    h["labels_txt"] = labels_for(a.kb, h)
                select(hits, a.provider, a.max_images, a.max_text, a.with_page)
            rep = tm.report()
            totals.append(rep["program"] * 1000)
            for s in rep["stages"]:
                per.setdefault(s["name"], []).append(s["ms"])
    totals.sort()
    print(f"\n{len(totals)} question runs on {key}, {a.retriever} + {a.reranker if a.reranker != 'gemini' else 'none (gemini rerank is an api call)'}")
    print(f"our code per question: median {statistics.median(totals):.1f} ms, p95 {totals[int(len(totals) * .95) - 1]:.1f} ms, max {totals[-1]:.1f} ms")
    for n, v in per.items():
        print(f"  {n:18} median {statistics.median(v):6.1f} ms")
elif a.cmd == "eval":
    ensure(a.kb, key, a.provider)
    report(evals.run(a.kb, key, a.provider, q, not a.no_judge, a.only, a.split, progress=lambda i, n, qq: print(f"[{i}/{n}] {qq}"), workers=a.workers))
