# self-checking rag: grade evidence before answering, verify claims after. one cheap model call each
import json
import re

from .config import PROVIDERS, ROOT

GRADE = """Question: {question}

Evidence a search engine found:
{evidence}

1. Which evidence numbers contain information that helps answer the question? Count it even if the wording differs:
   abbreviations (CR = credits), ranges written 60-89 vs "60 to 89", table rows, map labels and positions all count.
   Leave out evidence that is only about the same general topic.
2. Can the question be fully answered from those relevant pieces alone?
3. If not, write a better search query that would find the missing information (short, keywords), else "".
Reply with only JSON: {{"relevant": [numbers], "answerable": true or false, "query": "..."}}"""

VERIFY = """Question: {question}
Answer to check: {answer}

Evidence:
{evidence}
{image_note}
Split the answer into its separate factual claims (skip phrases like "I could not find"). For each claim decide if the evidence,
text or attached images, clearly supports it. For directions and positions, check the label positions (x grows east/right,
y grows south/down) and the images. A claim not shown by the evidence is unsupported even if it sounds right.
Reply with only JSON: {{"claims": [{{"claim": "...", "supported": true or false, "evidence": [numbers], "note": "few words"}}]}}"""


def evidence_text(sent, cap=1500):
    """same amount of text the answer model sees, so the checker never judges from less than the answer had"""
    out = []
    for h in sent:
        labs = f"\n   labels: {h['labels_txt'][:1200]}" if h.get("labels_txt") else ""
        out.append(f"[{h['n']}] {h['doc']} p{h['page']} {h.get('label', '')}: {h['text'][:cap]}{labs}")
    return "\n".join(out)


def parse_json(raw):
    try:
        return json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
    except Exception:
        return None


def grade(question, sent, provider, tm):
    """text model reads question + evidence, says which pieces matter and whether it's answerable"""
    from .llm import chat
    raw, secs = chat(provider, PROVIDERS[provider]["text"], [GRADE.format(question=question, evidence=evidence_text(sent))], max_tokens=500)
    tm.model("grade evidence", secs)
    j = parse_json(raw)
    if not j:   # grader broke: don't block the answer, keep everything
        return {"relevant": [h["n"] for h in sent], "answerable": True, "query": "", "ok": False}
    ns = {h["n"] for h in sent}
    rel = [int(n) for n in j.get("relevant", []) if str(n).isdigit() and int(n) in ns]
    return {"relevant": rel, "answerable": bool(j.get("answerable", bool(rel))), "query": str(j.get("query") or "")[:200], "ok": True}


def verify(question, answer, sent, provider, tm, spatial=""):
    """claim by claim check. visual evidence goes in as the real crops, so the model looks again instead of trusting itself"""
    from .llm import chat
    vis = [h for h in sent if h["kind"] == "visual" and h.get("crop")][:PROVIDERS[provider]["max_images"]]
    note = ("Images attached, in order: " + ", ".join(f"[{h['n']}]" for h in vis) + ".") if vis else ""
    ev = evidence_text(sent) + ("\n" + spatial if spatial else "")
    content = [VERIFY.format(question=question, answer=answer, evidence=ev, image_note=note)]
    for h in vis:
        content += [f"image for [{h['n']}]", ROOT / h["crop"]]
    raw, secs = chat(provider, PROVIDERS[provider]["vision" if vis else "text"], content, max_tokens=900)
    tm.model("verify answer", secs)
    j = parse_json(raw)
    if not j:
        return []
    return [{"claim": str(c.get("claim", ""))[:300], "supported": bool(c.get("supported")), "evidence": [int(n) for n in c.get("evidence", []) if str(n).isdigit()],
             "note": str(c.get("note", ""))[:120]} for c in j.get("claims", []) if c.get("claim")]


def fix_prompt(bad):
    return ("- A checker found these claims NOT supported by the evidence. Drop or correct them, using only the evidence:\n"
            + "\n".join(f"  * {c['claim']} ({c['note']})" for c in bad) + "\n")


def confidence(sent, g, claims):
    """0..1. blends how well retrieval scored, how much evidence the grader kept, and how many claims checked out"""
    parts = []
    scores = [h.get("score") for h in sent if h.get("score") is not None]
    if scores:
        parts.append((max(scores), .3))
    if g:
        parts.append((len(g["relevant"]) / max(1, len(sent)) if g["answerable"] else 0.0, .3))
    if claims:
        parts.append((sum(c["supported"] for c in claims) / len(claims), .4))
    if not parts:
        return None
    return round(sum(v * w for v, w in parts) / sum(w for _, w in parts), 3)