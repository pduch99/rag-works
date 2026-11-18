# gold question editing + ai drafting. drafts are only suggestions: check every one against the page before keeping it
import json
import re

from . import kb as kbs
from .chunkers import GRIDS
from .config import PROVIDERS
from .llm import chat

TYPES = ["text_fact", "exact_term", "table", "chart_value", "diagram_label", "visual_location", "visual_spatial", "visual_route",
         "cross_page", "cross_doc", "comparison", "unanswerable"]
CELLS = [n for n, _ in GRIDS["3x3"]]

DRAFT = """You are helping build a gold-standard test set for a document question-answering system.
This is page {page} of "{doc}". Its text layer is below (may be empty for scanned or image-only pages).
<text>
{text}
</text>
Write {n} questions a real reader might ask that this page answers. Mix kinds: at least one about something you can only
get from the picture/figure/table if there is one, and one exact detail (a number, code or name). Each answer must be
fully supported by this page.
For each, say where on the page the answer is, using a 3x3 grid: nw n ne / w c e / sw s se (one or more cells), or "page".
Reply with only JSON: {{"items": [{{"question": "...", "answer": "...", "key_facts": ["..."], "type": "one of {types}",
"cells": ["ne"], "quote": "exact text from the text layer that contains the answer, or empty"}}]}}"""


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def save_gold(kb, items):
    (kbs.kb_dir(kb) / "gold.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")


def cells_box(cells):
    boxes = [b for n, b in GRIDS["3x3"] if n in cells]
    if not boxes:
        return None
    return [round(min(b[0] for b in boxes), 4), round(min(b[1] for b in boxes), 4), round(max(b[2] for b in boxes), 4), round(max(b[3] for b in boxes), 4)]


def next_id(items, prefix):
    nums = [int(m.group(1)) for i in items if (m := re.match(rf"{re.escape(prefix)}-(\d+)$", i.get("id", "")))]
    return f"{prefix}-{max(nums, default=0) + 1:02d}"


def draft(kb, provider, path, page_no, n=3):
    """ask the model for candidate questions about one page"""
    pages = kbs.pages(kb, path)
    pg = pages[page_no - 1]
    prompt = DRAFT.format(page=page_no, doc=path.name, text=pg["text"][:5000], n=n, types=", ".join(TYPES[:-1]))
    content = [prompt] + ([pg["image"]] if pg["image"] else [])
    raw, _ = chat(provider, PROVIDERS[provider]["vision" if pg["image"] else "text"], content, max_tokens=1500)
    try:
        items = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))["items"]
    except Exception:
        return []
    out = []
    for it in items:
        cells = [c for c in it.get("cells", []) if c in CELLS]
        ev = {"doc": path.name, "page": page_no}
        quote = (it.get("quote") or "").strip()
        if quote and norm(quote) in norm(pg["text"]):   # only keep quotes that really are on the page
            ev["quote"] = quote
        elif cells and "page" not in it.get("cells", []):
            ev["box"] = cells_box(cells)
        out.append({"type": it.get("type") if it.get("type") in TYPES else "text_fact", "question": it.get("question", "").strip(),
                    "expected_answer": it.get("answer", "").strip(), "key_facts": [str(f) for f in it.get("key_facts", [])][:4],
                    "answerable": True, "evidence": [ev]})
    return [o for o in out if o["question"] and o["expected_answer"]]
