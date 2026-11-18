# spatial facts: small models misread label numbers and flip directions, so the code does the comparing.
# find the places the question names among the page's labels, then write "a is south-east of b" sentences from their positions
import math
import re

from . import kb as kbs
from . import layout as lay

STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "is", "are", "where", "what", "which", "how", "from",
        "relative", "located", "location", "map", "campus", "based", "does", "do", "get", "go", "near", "closest", "building",
        "buildings", "street", "there", "it", "its", "with", "by", "for", "between", "connect", "connects", "find"}
DIRS = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]


def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 1]


def direction(a, b):
    """where a is, seen from b. up = north, y grows south"""
    dx, dy = a["x"] - b["x"], b["y"] - a["y"]
    if abs(dx) < .02 and abs(dy) < .02:
        return "right next to"
    return DIRS[round(math.degrees(math.atan2(dy, dx)) / 45) % 8] + " of"


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def page_labels(kb, sent):
    """labels of every pdf page that has visual evidence, legend already left out, same text kept once per spot"""
    out = []
    for doc, page in sorted({(h["doc"], h["page"]) for h in sent if h["kind"] == "visual"}):
        path = kbs.kb_dir(kb) / "docs" / doc
        if path.exists() and kbs.kind(path) == "pdf":
            out += [{**l, "doc": doc, "page": page} for l in lay.analyze(path, page)["labels"]]
    return out


def named(question, labs):
    """labels the question talks about: most of the label's words (or its short code like SC) show up in the question"""
    q = set(words(question))
    found = []
    for l in labs:
        lw = words(l["text"])
        if not lw:
            continue
        hit = sum(w in q for w in lw)
        code = re.search(r"\(([A-Z]{2,4})\)", l["text"])
        if (code and code.group(1).lower() in q) or (hit >= 2 and hit / len(lw) >= .5) or (len(lw) == 1 and hit == 1 and len(lw[0]) >= 5):
            found.append((hit / len(lw), l))
    found.sort(key=lambda f: -f[0])
    keep = []
    for _, l in found:
        if not any(k["text"] == l["text"] and dist(k, l) < .05 for k in keep):
            keep.append(l)
    # order by where the question names them, so "ramp relative to the student center" reads "ramp is south-east of student center"
    ql = question.lower()
    first = lambda l: min((ql.find(w) for w in words(l["text"]) if ql.find(w) >= 0), default=len(ql))
    return sorted(keep[:4], key=first)


def facts(kb, question, sent, near=5):
    """plain sentences for the prompt, or "" when the question names no labeled place"""
    labs = page_labels(kb, sent)
    places = named(question, labs)
    if not places:
        return ""
    pos = lambda l: f"{l['text'][:45]} ({l['x'] * 100:.0f},{l['y'] * 100:.0f})"
    lines = []
    for i, a in enumerate(places):
        for b in places[i + 1:]:
            if a["text"] != b["text"]:
                lines.append(f"- {pos(a)} is {direction(a, b)} {pos(b)}")
    for a in places[:2]:
        others, seen = sorted((l for l in labs if l["text"] != a["text"] and l["page"] == a["page"]), key=lambda l: dist(l, a)), set()
        close = []
        for l in others:
            if l["text"] in seen or re.match(r"revision|date|\*", l["text"], re.I):
                continue
            seen.add(l["text"])
            close.append(f"{l['text'][:35]} is {direction(l, a)} it")
            if len(close) == near:
                break
        lines.append(f"- around {a['text'][:45]}: " + "; ".join(close))
    return ("Spatial facts (computed by code from the exact label positions, up = north. "
            "use these for directions instead of comparing numbers yourself):\n" + "\n".join(lines))
