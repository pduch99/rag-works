# gold-standard evals. retrieval and answer are scored separately (the fp5 lesson)
# gold evidence is {doc, page, box, quote}, not chunk ids, so every chunker is scored against the same truth
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import kb as kbs
from .answer import DEFAULTS, ask
from .config import PROVIDERS, RUNS, parse_ix
from .index import box_of
from .layout import area, overlap
from .llm import chat

QUAD = {"nw": (0, 0, .5, .5), "ne": (.5, 0, 1, .5), "sw": (0, .5, .5, 1), "se": (.5, .5, 1, 1), "full": (0, 0, 1, 1)}
GRADE_SCORE = {"correct": 1, "partial": .5, "wrong": 0}
STOP = set("a an the of to in on at for is are was were be by and or with from as it its this that".split())

JUDGE = """You are grading a RAG system's answer against a gold-standard answer.

Question: {question}
Gold answer: {expected}
{facts}
System answer: {answer}

Evidence the system was given:
{evidence}

Grade:
- grade: "correct" if it matches the gold answer's key facts, "partial" if some key facts are right and none contradict it, "wrong" otherwise.
  If the gold answer says the documents do not contain the answer, "correct" means the system said it could not find it, and "wrong" means it gave an answer anyway.
- grounded: "yes" if every claim in the system answer is supported by the evidence above, "no" if it adds or invents anything
Reply with only JSON: {{"grade": "...", "grounded": "...", "reason": "<one sentence>"}}"""


def normalize(g):
    """gold v1 (expected_docs/pages/regions) -> v2 (evidence list). fills defaults"""
    g = dict(g)
    if "evidence" not in g:
        ev = []
        for d in g.get("expected_docs", []):
            for p in g.get("expected_pages", {}).get(d, [1]):
                regs = g.get("expected_regions") or []
                ev += [{"doc": d, "page": p, "box": list(QUAD[r])} for r in regs] or [{"doc": d, "page": p}]
        g["evidence"] = ev
    g.setdefault("key_facts", [])
    g.setdefault("answerable", bool(g["evidence"]))
    g.setdefault("split", "dev")
    g.setdefault("type", "")
    return g


def gold(kb, split="all"):
    return [g for g in map(normalize, kbs.gold(kb)) if split in ("all", g["split"])]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def cover(ev, chunks):
    """0..1: how well these chunks cover one piece of gold evidence"""
    same = [c for c in chunks if c["doc"] == ev["doc"] and (ev.get("page") is None or c["page"] == ev["page"])]
    if ev.get("quote"):
        q = norm(ev["quote"])
        return 1.0 if any(q in norm(c["text"]) for c in same) else 0.0
    if ev.get("box"):
        b = tuple(ev["box"])
        got = sum(overlap(b, box_of(c)) for c in same if box_of(c))
        return round(min(1, got / max(area(b), 1e-9)), 3)
    return 1.0 if same else 0.0


def relevant(c, evidence):
    """is this one chunk relevant to any gold evidence (for mrr)"""
    for ev in evidence:
        if c["doc"] != ev["doc"] or (ev.get("page") is not None and c["page"] != ev["page"]):
            continue
        if ev.get("quote"):
            if norm(ev["quote"]) in norm(c["text"]):
                return True
        elif ev.get("box"):
            cb, b = box_of(c), tuple(ev["box"])
            if cb and overlap(cb, b) / max(min(area(cb), area(b)), 1e-9) >= .5:
                return True
        else:
            return True
    return False


def retrieval_scores(g, hits, sent):
    ev = g["evidence"]
    if not ev:   # unanswerable: nothing to retrieve
        return {"retrieval_pass": True, "evidence_sent": None, "recall@1": None, "recall@3": None, "recall@5": None, "mrr": None}
    s = {f"recall@{k}": round(sum(cover(e, hits[:k]) >= .5 for e in ev) / len(ev), 3) for k in (1, 3, 5)}
    first = next((h["rank"] for h in hits if relevant(h, ev)), None)
    s["mrr"] = round(1 / first, 3) if first else 0.0
    covers = [cover(e, sent) for e in ev]
    s["evidence_cover"] = [round(c, 2) for c in covers]
    s["evidence_sent"] = round(sum(c >= .5 for c in covers) / len(ev), 3)
    s["retrieval_pass"] = all(c >= .5 for c in covers)
    return s


def key_facts(g, answer):
    """share of key facts whose words all show up in the answer. cheap check next to the judge"""
    if not g["key_facts"]:
        return None
    a = set(norm(answer).split())
    hit = sum(all(w in a for w in norm(f).split() if w not in STOP) for f in g["key_facts"])
    return round(hit / len(g["key_facts"]), 3)


def judge(provider, g, r):
    # the judge sees what the answer model saw: text, label positions, spatial facts
    evidence = "\n".join(f"[{h['n']}] {h['text'][:1500]}" + (f"\n   labels: {h['labels_txt'][:1200]}" if h.get("labels_txt") else "")
                         for h in r["sent"]) or "(none)"
    if r.get("spatial"):
        evidence += "\n" + r["spatial"]
    facts = ("Key facts: " + "; ".join(g["key_facts"])) if g["key_facts"] else ""
    prompt = JUDGE.format(question=g["question"], expected=g["expected_answer"], facts=facts, answer=r["answer"], evidence=evidence)
    try:
        raw, _ = chat(provider, PROVIDERS[provider]["text"], [prompt], max_tokens=1200)
        j = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
        grade, grounded = j.get("grade", "").lower(), j.get("grounded", "").lower()
        grade = grade if grade in GRADE_SCORE else ""
        return {"grade": grade, "judge_grade": grade, "grounded": grounded if grounded in ("yes", "no") else "",
                "reason": j.get("reason", ""), "graded_by": "judge"}
    except Exception as e:
        return {"grade": "", "judge_grade": "", "grounded": "", "reason": f"judge failed: {e}", "graded_by": ""}


def diagnose(item):
    """which half of the pipeline broke. retrieval_pass decides who gets the blame"""
    ok, grade = item["retrieval_pass"], item.get("grade")
    if not grade:
        return "ungraded"
    if not item.get("answerable", True):
        return "pass: abstained" if grade == "correct" else "fail: hallucinated"
    if item.get("abstained"):
        return "fail: abstained" + (" (evidence was sent)" if ok else " (retrieval)")
    if grade == "correct":
        return "pass" if ok else "pass, weak retrieval"
    return f"{'fail' if grade == 'wrong' else 'partial'}: {'generation' if ok else 'retrieval'}"


def citation_ok(g, r):
    docs = {e["doc"] for e in g["evidence"]}
    return any(c["doc"] in docs for c in r["citations"]) if docs else None


def slim(h, keys):
    return {k: h.get(k) for k in keys}


HIT_KEYS = ("rank", "id", "doc", "page", "kind", "sub", "region", "label", "chunk", "box", "score", "distance", "vec_rank", "bm25_rank",
            "fused_rank", "moved", "sent", "n", "crop", "text")


EVAL_WORKERS = int(os.getenv("EVAL_WORKERS", 2))   # questions answered at the same time. lower it if you hit rate limits


def one(kb, key, provider, q, use_judge, g):
    base = {"id": g["id"], "question": g["question"], "type": g["type"], "split": g["split"], "answerable": g["answerable"],
            "expected_answer": g["expected_answer"]}
    try:
        r = ask(kb, key, g["question"], provider, q)
    except Exception as e:
        return {**base, "error": f"{type(e).__name__}: {e}", "retrieval_pass": False, "grade": "", "grounded": "", "diagnosis": "error"}
    chk = r.get("check") or {}
    item = {**base, "answer": r["answer"], "why": r["why"], "citations": r["citations"],
            "hits": [{**slim(h, HIT_KEYS), "text": h["text"][:300]} for h in r["hits"]],
            "sent": [slim(h, ("n", "id", "doc", "page", "kind", "region", "label", "crop", "box")) for h in r["sent"]],
            "seconds": r["seconds"], "program_seconds": r["timing"]["program"], "model_seconds": r["timing"]["model"],
            "llm_calls": r.get("llm_calls"), "model": r["model"], "abstained": chk.get("abstained", False),
            "confidence": chk.get("confidence"), "corrected": chk.get("corrected", False), "retried": chk.get("retried", False),
            "claims": chk.get("claims"), "original_answer": chk.get("original")}
    item.update(retrieval_scores(g, r["hits"], r["sent"]))
    item["key_facts"] = key_facts(g, r["answer"])
    item["citation_ok"] = citation_ok(g, r)
    if use_judge:
        item.update(judge(provider, g, r))
        item["llm_calls"] = (item["llm_calls"] or 0) + 1
    else:
        item.update({"grade": "", "judge_grade": "", "grounded": "", "reason": "", "graded_by": ""})
    item["diagnosis"] = diagnose(item)
    return item


def run(kb, key, provider, q=None, use_judge=True, only=None, split="all", progress=None, note="", workers=None):
    """answers the gold questions a few at a time (the api is the slow part, so overlap the waiting)"""
    q = {**DEFAULTS, **(q or {})}
    items_gold = [g for g in gold(kb, split) if not only or g["id"] in only]
    t0 = time.time()
    done, results = 0, {}
    with ThreadPoolExecutor(workers or EVAL_WORKERS) as pool:
        futs = {pool.submit(one, kb, key, provider, q, use_judge, g): g for g in items_gold}
        for f in as_completed(futs):
            g = futs[f]
            results[g["id"]] = f.result()
            done += 1
            if progress:   # called from this thread, so streamlit progress bars work
                progress(done, len(items_gold), g["question"])
    items = [results[g["id"]] for g in items_gold]
    out = {"kb": kb, "index": key, **parse_ix(key), "strategy": parse_ix(key)["chunker"], "provider": provider,
           "vision_model": PROVIDERS[provider]["vision"], "settings": q, "split": split, "note": note,
           "at": time.strftime("%Y-%m-%d %H:%M"), "wall_seconds": round(time.time() - t0, 1), "items": items}
    out["summary"] = summarize(items)
    chk = f"__{q['check']}" if q.get("check", "none") != "none" else ""
    lab = ("" if q.get("labels", True) else "__nolabels") + ("__spatial" if q.get("spatial") else "")
    name = f"{time.strftime('%Y%m%d-%H%M%S')}__{key}__{q['retriever']}-{q['reranker']}{chk}{lab}__{provider}{'__page' if q['with_page'] else ''}.json"
    save(kb, name, out)
    out["file"] = name
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(items):
    n = len(items) or 1
    graded = [i for i in items if i.get("grade")]
    ans = [i for i in items if i.get("answerable", True)]
    unans = [i for i in items if not i.get("answerable", True)]
    checked = [i for i in items if i.get("checked") and i.get("judge_grade")]
    s = {
        "questions": len(items),
        "retrieval_pass": round(sum(i["retrieval_pass"] for i in ans) / (len(ans) or 1), 3),
        "recall@1": mean(i.get("recall@1") for i in ans), "recall@3": mean(i.get("recall@3") for i in ans),
        "recall@5": mean(i.get("recall@5") for i in ans), "mrr": mean(i.get("mrr") for i in ans),
        "evidence_sent": mean(i.get("evidence_sent") for i in ans),
        "answer_score": mean(GRADE_SCORE[i["grade"]] for i in graded),
        "key_facts": mean(i.get("key_facts") for i in items),
        "correct": sum(i.get("grade") == "correct" for i in items),
        "partial": sum(i.get("grade") == "partial" for i in items),
        "wrong": sum(i.get("grade") == "wrong" for i in items),
        "grounded": mean((i["grounded"] == "yes") * 1.0 for i in items if i.get("grounded")),
        "citation_ok": mean((i["citation_ok"] * 1.0) for i in items if i.get("citation_ok") is not None),
        "abstain_ok": mean((i.get("grade") == "correct") * 1.0 for i in unans if i.get("grade")),
        "generation_failures": sum(i.get("diagnosis", "").endswith(": generation") for i in items),
        "retrieval_failures": sum(i.get("diagnosis", "").endswith(": retrieval") for i in items),
        "hallucinated": sum(i.get("diagnosis") == "fail: hallucinated" for i in items),
        "avg_seconds": round(sum(i.get("seconds", 0) for i in items) / n, 2),
        "graded": len(graded),
        "judge_agreement": mean((i["grade"] == i["judge_grade"]) * 1.0 for i in checked), "checked": len(checked),
        "false_abstain": round(sum(bool(i.get("abstained")) for i in ans) / (len(ans) or 1), 3),
        "corrected": sum(bool(i.get("corrected")) for i in items),
        "confidence": mean(i.get("confidence") for i in items),
        "llm_calls": mean(i.get("llm_calls") for i in items),
        "program_ms": round(1000 * (mean(i.get("program_seconds") for i in items) or 0), 1),
        "model_seconds": mean(i.get("model_seconds") for i in items),
    }
    s["region_recall"] = s["evidence_sent"]   # old name, older runs + charts
    by = {}
    for i in items:
        by.setdefault(i.get("type") or "?", []).append(i)
    s["by_type"] = {t: {"n": len(v), "retrieval_pass": round(sum(x["retrieval_pass"] for x in v) / len(v), 3),
                        "recall@5": mean(x.get("recall@5") for x in v),
                        "answer_score": mean(GRADE_SCORE[x["grade"]] for x in v if x.get("grade"))} for t, v in sorted(by.items())}
    return s


def runs_dir(kb):
    d = RUNS / kb
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(kb, name, data):
    data = {k: v for k, v in data.items() if k != "file"}
    (runs_dir(kb) / name).write_text(json.dumps(data, indent=1), encoding="utf-8")


def list_runs(kb):
    return sorted((p.name for p in runs_dir(kb).glob("*.json")), reverse=True)


_runs = {}


def load(kb, name):
    p = runs_dir(kb) / name
    k = (str(p), p.stat().st_mtime_ns)
    if k not in _runs:
        _runs[k] = _load(kb, name)
    return json.loads(json.dumps(_runs[k]))   # copy so callers can't change the cache


def _load(kb, name):
    d = json.loads((runs_dir(kb) / name).read_text(encoding="utf-8"))
    d["file"] = name
    d.setdefault("index", d.get("strategy", "?"))
    d.setdefault("settings", {})
    d["settings"].setdefault("labels", False)   # runs from before fp9 had no label positions
    for k2, v in DEFAULTS.items():
        d["settings"].setdefault(k2, v)
    if "by_type" not in d.get("summary", {}):
        d["summary"] = summarize(d["items"])
    return d


def label(r):
    """short name for a run, used in charts"""
    s = r["settings"]
    return (f"{r['index']} · {s['retriever']}" + (f"+{s['reranker']}" if s["reranker"] != "none" else "")
            + (f" · check {s['check']}" if s.get("check", "none") != "none" else "") + ("" if s.get("labels", True) else " · no labels") + (" · spatial" if s.get("spatial") else "")
            + (" · +page" if s["with_page"] else "") + f" · {r['provider']}")


def regrade(kb, name, grades):
    """manual grades from the ui. grades = {id: {"grade":.., "grounded":.., "checked": bool}}"""
    d = load(kb, name)
    for item in d["items"]:
        g = grades.get(item["id"])
        if not g:
            continue
        changed = g.get("grade", item.get("grade")) != item.get("grade") or g.get("grounded", item.get("grounded")) != item.get("grounded")
        item.update({k: v for k, v in g.items() if k in ("grade", "grounded", "checked")})
        if changed:
            item["graded_by"] = "manual"
            item["checked"] = True
        item["diagnosis"] = diagnose(item)
    d["summary"] = summarize(d["items"])
    save(kb, name, d)
    return d
