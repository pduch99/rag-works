# question -> retrieve -> pick evidence -> (check) -> model -> (verify) -> cited answer. everything it did is in the returned dict
import re

from . import kb as kbs
from . import layout as lay
from .check import confidence, fix_prompt, grade, verify
from .config import PROVIDERS, ROOT
from .index import box_of
from .llm import chat
from .retrieve import search
from .spatial import facts as spatial_facts
from .timing import Timing

PROMPT = """Answer the question using ONLY the numbered evidence below.

{evidence}
{image_note}
Rules:
- Put the evidence number in brackets after each fact you use, like [1] or [2][3].
- Each crop says where it sits on its page (top-left, figure 2, etc). Use that when comparing positions.
- When label positions are listed, they are exact page positions: x grows to the right (east), y grows downward (south).
  To compare two places, look up both labels and compare their numbers. A smaller y is further north.
- For maps, use the compass if one is shown. Otherwise up is north.
- If the evidence does not answer the question, say you could not find it. Do not guess.
{extra}
Reply in exactly this format:
ANSWER: <short direct answer with [n] citations>
WHY: <one or two sentences on which evidence shows it>

Question: {question}"""

DEFAULTS = {"retriever": "vector", "reranker": "none", "k": 8, "pool": 20, "max_images": 2, "max_text": 2, "with_page": False,
            "labels": True, "check": "none", "spatial": False}
ABSTAIN = "I couldn't find this in the documents."


def label(h):
    """citation label like campus_map.pdf p1 NE"""
    s = f"{h['doc']} p{h['page']}"
    if h["kind"] == "visual":
        return s + " " + (h.get("label") or h["region"].upper())
    return s + (" table" if h.get("sub") == "table" else "")


def labels_for(kb, h, limit=70):
    """text-layer labels inside this crop, with page positions as 0-100. empty for images, text files and scans"""
    path = kbs.kb_dir(kb) / "docs" / h["doc"]
    if h["kind"] != "visual" or not path.exists() or kbs.kind(path) != "pdf":
        return ""
    b = box_of(h) or (0, 0, 1, 1)
    labs = [x for x in lay.analyze(path, h["page"])["labels"] if b[0] <= x["x"] <= b[2] and b[1] <= x["y"] <= b[3]]
    if not labs:
        return ""
    labs.sort(key=lambda x: (round(x["y"], 2), x["x"]))
    return "; ".join(f"{x['text'][:40]} ({x['x'] * 100:.0f},{x['y'] * 100:.0f})" for x in labs[:limit])


def select(hits, provider, max_images=2, max_text=2, with_page=False):
    """retrieval can look at lots of candidates, the model only gets the best few that fit the budget"""
    p = PROVIDERS[provider]
    visual = [h for h in hits if h["kind"] == "visual"][:min(max_images, p["max_images"])]
    text = [h for h in hits if h["kind"] == "text"][:max_text]
    page = None
    if with_page and visual and visual[0]["box"] != "0.0000,0.0000,1.0000,1.0000":
        page = visual[0]["page_image"]   # whole page for orientation, uses one image slot
        if len(visual) + 1 > p["max_images"]:
            visual = visual[:p["max_images"] - 1]
    sent = sorted(visual + text, key=lambda h: h["rank"])
    # trim to the provider's token budget: drop lowest ranked text, then shorten everything
    images = len(visual) + (1 if page else 0)
    cap = 1500

    def est():
        return images * p["image_tokens"] + 500 + sum(min(len(h["text"]), cap) + len(h.get("labels_txt", "")) for h in sent) // 4
    while est() > p["input_budget"] and any(h["kind"] == "text" for h in sent) and len(sent) > 1:
        drop = max((h for h in sent if h["kind"] == "text"), key=lambda h: h["rank"])
        sent.remove(drop)
    while est() > p["input_budget"] and cap > 300:
        cap -= 200
    return sent, page, cap


def number(hits, sent):
    for n, h in enumerate(sent, start=1):
        h["n"] = n
    ids = {h["id"] for h in sent}
    for h in hits:
        h["sent"] = h["id"] in ids
        if not h["sent"]:
            h.pop("n", None)


def build_prompt(question, sent, page, cap, extra="", spatial=""):
    evidence, img_lines = [], []
    for h in sent:
        where = f", {h.get('where') or h['region']}" if h["kind"] == "visual" else (" (table)" if h.get("sub") == "table" else "")
        kind = "image description" if h["kind"] == "visual" else "text"
        labs = f"\nLabel positions on the page (x,y in % from top-left): {h['labels_txt']}" if h.get("labels_txt") else ""
        evidence.append(f"[{h['n']}] {h['doc']}, page {h['page']}{where}, {kind}:\n{h['text'][:cap]}{labs}\n")
        if h["kind"] == "visual":
            img_lines.append(f"image for [{h['n']}]")
    note = ""
    if img_lines or page:
        order = (["the whole page (for orientation, not citable)"] if page else []) + img_lines
        note = "The images attached after this text are, in order: " + "; ".join(order) + ".\n"
    if spatial and evidence:
        evidence.append(spatial + "\n")
    prompt = PROMPT.format(evidence="\n".join(evidence) if evidence else "(no evidence found)", image_note=note, question=question, extra=extra)
    content = [prompt] + ([ROOT / page] if page else [])
    for h in sent:
        if h["kind"] == "visual":
            content += [f"image for [{h['n']}]", ROOT / h["crop"]]
    return prompt, content, bool(img_lines or page)


def ask(kb, key, question, provider, q=None):
    """key = index config (chunker.embedder.style[.ctx]), q = query settings (see DEFAULTS)"""
    q = {**DEFAULTS, **(q or {})}
    tm = Timing()
    calls = 0
    chk = {"mode": q["check"], "grade": None, "claims": None, "corrected": False, "original": None, "abstained": False, "retried": False}

    def retrieve_and_select(query, pool):
        hits = search(kb, key, query, q["retriever"], q["reranker"], q["k"], pool, provider, tm)
        with tm.stage("labels + select"):
            for h in hits:
                h["labels_txt"] = labels_for(kb, h) if q["labels"] else ""
            sent, page, cap = select(hits, provider, q["max_images"], q["max_text"], q["with_page"])
        return hits, sent, page, cap
    hits, sent, page, cap = retrieve_and_select(question, q["pool"])
    calls += q["reranker"] == "gemini"
    # grade the evidence before answering: drop what's irrelevant, retry once with a better query, abstain if nothing helps
    if q["check"] in ("grade", "full", "auto") and sent:
        number(hits, sent)
        g = grade(question, sent, provider, tm)
        calls += 1
        if not g["relevant"]:   # retry only when nothing helped. a small grader saying "not answerable" too often made retries swap good evidence for worse
            hits2, sent2, page2, cap2 = retrieve_and_select(g["query"] or question, q["pool"] * 2)
            calls += q["reranker"] == "gemini"
            number(hits2, sent2)
            g2 = grade(question, sent2, provider, tm)
            calls += 1
            chk["retried"] = True
            if len(g2["relevant"]) > len(g["relevant"]) or (g2["answerable"] and not g["answerable"]):
                hits, sent, page, cap, g = hits2, sent2, page2, cap2, g2
        chk["grade"] = g
        # only refuse when the grader found nothing relevant at all. if some evidence is relevant, answer from it and
        # let the answer model say "not found" itself (a strict grader caused false "not found"s on answerable questions)
        if g["relevant"]:
            # the grader can drop pieces, but never the top hit or the top image: a small grader threw away the right crop
            keep = set(g["relevant"]) | {sent[0]["n"]} | {next((h["n"] for h in sent if h["kind"] == "visual"), None)}
            sent = [h for h in sent if h["n"] in keep]
            if page and not any(h["kind"] == "visual" for h in sent):
                page = None
        else:
            chk["abstained"] = True
    number(hits, sent)
    spatial = ""
    if q.get("spatial") and not chk["abstained"] and any(h["kind"] == "visual" for h in sent):
        with tm.stage("spatial facts"):
            spatial = spatial_facts(kb, question, sent)
    if chk["abstained"] or not sent:
        chk["abstained"] = True
        closest = ", ".join(label(h) for h in sent[:3]) or "nothing relevant was retrieved"
        raw, model, prompt = f"ANSWER: {ABSTAIN}\nWHY: The closest evidence ({closest}) does not answer it.", "(no model call)", ""
    else:
        prompt, content, visual = build_prompt(question, sent, page, cap, spatial=spatial)
        model = PROVIDERS[provider]["vision" if visual else "text"]
        raw, secs = chat(provider, model, content)
        tm.model("answer", secs)
        calls += 1
    with tm.stage("parse"):
        answer, why = parse(raw)
    # verify the answer claim by claim, fix it once if something isn't backed by the evidence
    conf_now = confidence(sent, chk["grade"], None)
    if not chk["abstained"] and (q["check"] in ("verify", "full") or (q["check"] == "auto" and conf_now < .75)):
        claims = verify(question, answer, sent, provider, tm, spatial)
        calls += 1
        chk["claims"] = claims
        bad = [c for c in claims if not c["supported"]]
        if bad:
            prompt2, content2, visual = build_prompt(question, sent, page, cap, fix_prompt(bad), spatial)
            raw2, secs = chat(provider, PROVIDERS[provider]["vision" if visual else "text"], content2)
            tm.model("fix answer", secs)
            calls += 1
            chk.update(corrected=True, original=answer)
            raw = raw2
            answer, why = parse(raw2)
    chk["confidence"] = None if chk["abstained"] else confidence(sent, chk["grade"], chk["claims"])
    with tm.stage("citations"):
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer + " " + why)})
        citations = [{"n": h["n"], "label": label(h), "id": h["id"], "kind": h["kind"], "crop": h["crop"], "doc": h["doc"],
                      "page": h["page"], "region": h["region"]} for h in sent if h["n"] in cited]
    t = tm.report()
    return {"question": question, "answer": answer, "why": why, "raw": raw, "citations": citations, "hits": hits,
            "sent": sent, "spatial": spatial, "page_image": page, "prompt": prompt, "model": model, "provider": provider, "index": key, "strategy": key,
            "settings": q, "check": chk, "llm_calls": calls, "timing": t,
            "retrieve_seconds": round(sum(s["ms"] for s in t["stages"] if not s["model"]) / 1000, 3),
            "model_seconds": t["model"], "seconds": t["total"]}


def parse(raw):
    if not raw.strip():
        return "(the model returned nothing. it may have spent its whole token budget thinking)", ""
    m = re.search(r"ANSWER:\s*(.*?)(?:\n\s*WHY:\s*(.*))?$", raw, flags=re.S | re.I)
    if not m:
        return raw.strip(), ""
    return m.group(1).strip(), (m.group(2) or "").strip()