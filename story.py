# the whole fp6 -> fp10 story in one go: builds what it needs, runs the evals, asks the demo questions, times our code,
# and writes story_report.txt (paste that back to claude for the walkthrough)
# run from the rag-works folder with the venv on and GEMINI_API_KEY in .env:
#   python story.py                      all 5 steps on metrostate (~20-30 min on the free tier, ~10 min on a local gpu)
#   python story.py --provider ollama    same thing on your own gpu, no limits
#   python story.py --kb testbed --split dev
#   python story.py --skip-evals         just the demo questions + timing (~2 min)
# runs land in runs/<kb>/ so the app's evals + compare views show them too
import argparse
import os
import platform
import statistics
import sys
import time
import traceback
from importlib import metadata

from rag import evals, index, kb as kbs
from rag.answer import ask, labels_for, select
from rag.config import DEFAULT_PROVIDER, PROVIDERS, has_key, ix_key, parse_ix
from rag.retrieve import search
from rag.timing import Timing

# each step changes one stage from the step before
STEPS = [
    ("fp6 baseline", ("region", "minilm", "prose", False), {"retriever": "vector", "reranker": "none", "labels": False, "check": "none"},
     "old embeddings, 2x2 quadrants, prose image descriptions, vector search, answer straight away"),
    ("fp7 better index", ("layout", "bge", "structured", True), {"retriever": "vector", "reranker": "none", "labels": False, "check": "none"},
     "bge embeddings, structure + layout chunks, structured descriptions, contextual chunks"),
    ("fp8 hybrid + rerank", ("layout", "bge", "structured", True), {"retriever": "hybrid", "reranker": "cross", "labels": False, "check": "none"},
     "same index, keyword + vector search fused, cross-encoder rerank"),
    ("fp9 labels + self-check", ("layout", "bge", "structured", True), {"retriever": "hybrid", "reranker": "cross", "labels": True, "check": "auto"},
     "same retrieval, pdf label positions in the evidence, grade evidence + verify when unsure"),
    ("fp10 spatial facts", ("layout", "bge", "structured", True), {"retriever": "hybrid", "reranker": "cross", "labels": True, "check": "auto", "spatial": True},
     "same as fp9, plus directions between the places the question names, computed by code from the label positions"),
]
DEMO = [
    ("where is the library", "Where is the Library and Learning Center located?"),
    ("fp5 failure: parking ramp", "Where is the parking ramp relative to the Student Center?"),
    ("unanswerable", "How many parking spaces does the Parking Ramp have?"),
    ("exact term", "When can a student with 60 to 89 credits register for spring 2027?"),
]
METRICS = [("retrieval pass", "retrieval_pass"), ("recall@5", "recall@5"), ("mrr", "mrr"), ("answer score", "answer_score"),
           ("grounded", "grounded"), ("citation ok", "citation_ok"), ("key facts", "key_facts"), ("hallucinated", "hallucinated"),
           ("false not-found", "false_abstain"), ("corrected", "corrected"), ("gen failures", "generation_failures"),
           ("retr failures", "retrieval_failures"), ("calls/q", "llm_calls"), ("our code ms/q", "program_ms"), ("sec/q", "avg_seconds")]

p = argparse.ArgumentParser()
p.add_argument("--kb", default="metrostate")
p.add_argument("--split", default="all", choices=["all", "dev", "test"])
p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS))
p.add_argument("--skip-evals", action="store_true")
p.add_argument("--out", default="story_report.txt")
a = p.parse_args()

lines, errors = [], []


def out(s=""):
    print(s)
    lines.append(s)


def fmt(v):
    return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


failed_builds = {}


def ensure(key):
    if key in failed_builds:
        raise RuntimeError(f"index {key} already failed to build this run: {failed_builds[key]}")
    st = index.status(a.kb, key)
    if not st or st["stale"]:
        c = parse_ix(key)
        print(f"building {key} (first time can take a few minutes)")
        try:
            st = index.build(a.kb, c["chunker"], a.provider, c["embedder"], c["style"], c["contextual"], log=lambda m: print("  " + m))
        except Exception as e:
            failed_builds[key] = str(e)[:200]
            raise
    return st


out("RAG WORKS STORY REPORT")
out(f"made {time.strftime('%Y-%m-%d %H:%M')} · kb {a.kb} · split {a.split} · provider {a.provider}")
out(f"python {platform.python_version()} · {platform.system()} {platform.release()}")
pk = []
for name in ("streamlit", "chromadb", "fastembed", "rank-bm25", "pymupdf", "openai", "groq"):
    try:
        pk.append(f"{name} {metadata.version(name)}")
    except Exception:
        pk.append(f"{name} ?")
out("packages: " + ", ".join(pk))
out(f"models: vision {PROVIDERS[a.provider]['vision']} · text {PROVIDERS[a.provider]['text']}")
out(f"workers: EVAL_WORKERS={os.getenv('EVAL_WORKERS', '2')} LLM_WORKERS={os.getenv('LLM_WORKERS', '3')}")
gold = evals.gold(a.kb, a.split)
out(f"docs: {', '.join(d.name for d in kbs.docs(a.kb))}")
out(f"gold questions: {len(gold)} ({', '.join(sorted({g['type'] for g in gold}))})")
if not has_key(a.provider):
    out(f"\nSTOP: no {PROVIDERS[a.provider]['key']} in .env")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    sys.exit(1)
# one tiny call to each model first. if the api isn't working there's no point grinding through 150 calls
from rag.llm import ping
out("\napi check:")
bad = False
for m in sorted({PROVIDERS[a.provider]["text"], PROVIDERS[a.provider]["vision"]}):
    ok, msg = ping(a.provider, m)
    out(("  ok   " if ok else "  FAIL ") + msg)
    bad = bad or not ok
if bad:
    out("\nSTOP: the model api isn't answering, so every eval would fail. what the message above usually means:")
    out("  'limit: 0' / 'quota' -> this model has no free quota on your key. set another model in .env, or try --provider groq")
    out("  'not found' / 404 -> the model name is wrong or retired. check the names in google ai studio")
    out("  503 / overloaded -> google's side is busy. wait a few minutes and run again, or use --provider groq")
    out("  'per day' -> daily free limit used up. it resets tomorrow, or use --provider groq")
    out("  ollama 'connection error' -> ollama isn't running. open the ollama app, then run again")
    out("  ollama 'not found' -> model not pulled yet. run: ollama pull qwen3-vl:8b")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nwrote {a.out}")
    sys.exit(1)

# 1. the eval ladder
results = []
if not a.skip_evals:
    for step, ix, q, about in STEPS:
        key = ix_key(*ix)
        out(f"\n=== {step}: {about}")
        out(f"index {key} · {q}")
        try:
            st = ensure(key)
            out(f"index: {st['chunks']} chunks ({st.get('visual', '?')} visual), built {st['built']}")
            t0 = time.time()
            r = evals.run(a.kb, key, a.provider, q, True, None, a.split, note=f"story: {step}",
                          progress=lambda i, n, qq: print(f"  [{i}/{n}] {qq}"))
            out(f"saved runs/{a.kb}/{r['file']} · took {time.time() - t0:.0f}s")
            results.append((step, r))
            out("question   type             retr  r@5   answer   grounded diagnosis                 conf  calls  sec")
            for i in r["items"]:
                if i.get("error"):
                    out(f"{i['id']:10} ERROR {i['error'][:300]}")
                    errors.append(f"{step} {i['id']}: {i['error']}")
                    continue
                out(f"{i['id']:10} {i.get('type', '')[:16]:16} {'PASS' if i['retrieval_pass'] else 'FAIL'}  {fmt(i.get('recall@5')):5} "
                    f"{(i.get('grade') or '?'):8} {(i.get('grounded') or '?'):8} {i.get('diagnosis', '')[:25]:25} {fmt(i.get('confidence')):5} "
                    f"{fmt(i.get('llm_calls')):5} {fmt(i.get('seconds'))}")
            if all(i.get("error") for i in r["items"]):
                out("\nSTOP: every question in this step errored, so the api isn't usable right now. see the errors above")
                break
            for i in r["items"]:
                if i.get("answer"):
                    out(f"  {i['id']}: {i['answer'][:220]}" + (f"  [judge: {i.get('reason', '')[:120]}]" if i.get("reason") else ""))
        except Exception as e:
            out(f"STEP FAILED: {type(e).__name__}: {e}")
            errors.append(f"{step}: {traceback.format_exc()}")

# 2. side by side
if results:
    out("\n=== SUMMARY (each column changes one stage from the one before)")
    out(f"{'metric':18}" + "".join(f"{s[:22]:>24}" for s, _ in results))
    for lab, k in METRICS:
        out(f"{lab:18}" + "".join(f"{fmt(r['summary'].get(k)):>24}" for _, r in results))
    out("\nby question type (answer score / retrieval pass):")
    types = sorted({t for _, r in results for t in r["summary"].get("by_type", {})})
    for t in types:
        cells = []
        for _, r in results:
            v = r["summary"].get("by_type", {}).get(t)
            cells.append(f"{fmt(v['answer_score'])}/{fmt(v['retrieval_pass'])}" if v else "-")
        out(f"{t[:18]:18}" + "".join(f"{c:>24}" for c in cells))
    out("\nquestions that changed between steps:")
    for (s1, r1), (s2, r2) in zip(results, results[1:]):
        d1 = {i["id"]: i.get("diagnosis") for i in r1["items"]}
        ch = [f"{i['id']}: {d1.get(i['id'])} -> {i.get('diagnosis')}" for i in r2["items"] if d1.get(i["id"]) != i.get("diagnosis")]
        out(f"  {s1} -> {s2}: " + ("; ".join(ch) if ch else "no change"))

# 3. demo questions on the final config, with everything the walkthrough needs
step, ix, q, _ = STEPS[-1]
key = ix_key(*ix)
out(f"\n=== DEMO QUESTIONS on the {step} setup ({key}, {q['retriever']}+{q['reranker']}, check {q['check']})")
try:
    ensure(key)
    for tag, question in DEMO:
        out(f"\n--- {tag}: {question}")
        try:
            r = ask(a.kb, key, question, a.provider, q)
            c = r["check"]
            out(f"answer: {r['answer']}")
            out(f"why: {r['why']}")
            out("citations: " + (", ".join(f"[{x['n']}] {x['label']}" for x in r["citations"]) or "none"))
            out(f"check: confidence {fmt(c.get('confidence'))} · abstained {c['abstained']} · retried {c['retried']} · corrected {c['corrected']}"
                + (f" · grader kept {c['grade']['relevant']} answerable={c['grade']['answerable']} query='{c['grade']['query']}'" if c.get("grade") else ""))
            for cl in c.get("claims") or []:
                out(f"  {'OK ' if cl['supported'] else 'BAD'} {cl['claim']} {cl['evidence']} {cl['note']}")
            if c.get("original"):
                out(f"  before the fix: {c['original'][:300]}")
            out("top hits (rank. source | vec rank, bm25 rank, rerank move | sent):")
            for h in r["hits"][:6]:
                out(f"  {h['rank']}. {h['doc']} p{h['page']} {h.get('label', '')} | vec {h.get('vec_rank')}, bm25 {h.get('bm25_rank')}, moved {h.get('moved')}"
                    f" | {'sent [' + str(h['n']) + ']' if h.get('sent') else '-'}")
            for h in r["sent"]:
                if h.get("labels_txt"):
                    out(f"  label positions in [{h['n']}] {h.get('label', '')}: {h['labels_txt'][:400]}")
            if r.get("spatial"):
                out("  " + r["spatial"].replace("\n", "\n  "))
            t = r["timing"]
            out(f"time: {t['total']:.2f}s total = {t['program'] * 1000:.0f} ms our code + {t['model']:.2f}s model api · {r['llm_calls']} model calls")
            out("  " + " · ".join(f"{s['name']} {s['ms']:.0f}ms" + (" (api)" if s["model"] else "") for s in t["stages"]))
        except Exception as e:
            out(f"FAILED: {type(e).__name__}: {e}")
            errors.append(f"demo {tag}: {traceback.format_exc()}")
except Exception as e:
    out(f"DEMO FAILED: {type(e).__name__}: {e}")
    errors.append(f"demo: {traceback.format_exc()}")

# 4. our code only, no api: is the program ever the slow part?
out("\n=== SPEED: our code only, no api calls (3 rounds over the gold questions)")
try:
    qs = [g["question"] for g in gold] or [d[1] for d in DEMO]
    search(a.kb, key, qs[0], q["retriever"], q["reranker"], 8, 20)   # warm the models first
    tot, per = [], {}
    for _ in range(3):
        for qq in qs:
            tm = Timing()
            hits = search(a.kb, key, qq, q["retriever"], q["reranker"], 8, 20, a.provider, tm)
            with tm.stage("labels + select"):
                for h in hits:
                    h["labels_txt"] = labels_for(a.kb, h)
                select(hits, a.provider, 2, 3, False)
            rep = tm.report()
            tot.append(rep["program"] * 1000)
            for s in rep["stages"]:
                per.setdefault(s["name"], []).append(s["ms"])
    tot.sort()
    out(f"{len(tot)} runs: median {statistics.median(tot):.1f} ms, p95 {tot[max(0, int(len(tot) * .95) - 1)]:.1f} ms, max {tot[-1]:.1f} ms")
    for n, v in per.items():
        out(f"  {n:18} median {statistics.median(v):6.1f} ms")
except Exception as e:
    out(f"SPEED TEST FAILED: {type(e).__name__}: {e}")
    errors.append(f"speed: {traceback.format_exc()}")

out("\n=== ERRORS")
out("\n".join(errors) if errors else "none")
open(a.out, "w", encoding="utf-8").write("\n".join(lines))
print(f"\nwrote {a.out}. paste it back to claude for the walkthrough")
