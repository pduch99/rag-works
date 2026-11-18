# reads a pdf page's structure: text blocks with font info, tables, figure boxes. all boxes are 0..1 fractions of the page
import json
import re
import numpy as np
import pymupdf


def frac(rect, page):
    w, h = page.rect.width, page.rect.height
    x0, y0, x1, y1 = rect
    return (max(0, x0 / w), max(0, y0 / h), min(1, x1 / w), min(1, y1 / h))


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def inside(b, box, need=.6):
    return area(b) > 0 and overlap(b, box) / area(b) >= need


def blocks(page):
    """text blocks: text, box, font size, bold"""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = "\n".join(" ".join(s["text"] for s in line["spans"]).strip() for line in b["lines"]).strip()
        out.append({"text": text, "box": frac(b["bbox"], page), "size": max(s["size"] for s in spans),
                    "bold": all(s["flags"] & 16 for s in spans), "lines": len(b["lines"])})
    return out


def body_size(pdf):
    """most common font size across the doc, weighted by characters"""
    count = {}
    for page in pdf:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    k = round(s["size"], 1)
                    count[k] = count.get(k, 0) + len(s["text"].strip())
    return max(count, key=count.get) if count else 10


def is_heading(b, body):
    t = b["text"].strip()
    if not t or len(t) > 120 or b["lines"] > 3 or re.fullmatch(r"[\d\W]+", t):
        return False
    return b["size"] >= body * 1.18 or (b["bold"] and len(t) < 90 and not t.endswith((".", ",", ";")))


def tables(page):
    """[(box, markdown)] from pymupdf's table finder. skips fake tables like a map's street grid (huge or mostly empty)"""
    out = []
    try:
        for t in page.find_tables().tables:
            b = frac(t.bbox, page)
            cells = [c for row in t.extract() for c in row]
            filled = sum(1 for c in cells if c and str(c).strip()) / max(1, len(cells))
            md = t.to_markdown().strip()
            if md and len(md) > 20 and area(b) < .5 and filled >= .5:
                out.append((b, md))
    except Exception:
        pass
    return out


def figures(page, table_boxes=(), grid=160):
    """figure boxes: images + clusters of vector drawings. draws them on a small mask, grows it, finds blobs"""
    w, h = page.rect.width, page.rect.height
    gw, gh = grid, max(1, int(grid * h / w))
    mask = np.zeros((gh, gw), dtype=bool)
    rects = []
    try:
        rects += [i["bbox"] for i in page.get_image_info()]
    except Exception:
        pass
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None and not (r.width >= w * .95 and r.height >= h * .95):   # skip page frames / backgrounds
            rects.append(tuple(r))
    for r in rects:
        b = frac(r, page)
        if any(inside(b, t, .5) for t in table_boxes):
            continue
        x0, y0 = int(b[0] * gw), int(b[1] * gh)
        x1, y1 = max(x0 + 1, int(np.ceil(b[2] * gw))), max(y0 + 1, int(np.ceil(b[3] * gh)))
        mask[y0:y1, x0:x1] = True
    if not mask.any():
        return []
    for _ in range(2):   # grow so nearby bits join into one figure
        g = mask.copy()
        g[1:, :] |= mask[:-1, :]; g[:-1, :] |= mask[1:, :]; g[:, 1:] |= mask[:, :-1]; g[:, :-1] |= mask[:, 1:]
        mask = g
    seen = np.zeros_like(mask)
    boxes = []
    for y in range(gh):
        for x in range(gw):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, xs, ys = [(y, x)], [], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                xs.append(cx); ys.append(cy)
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            b = (min(xs) / gw, min(ys) / gh, (max(xs) + 1) / gw, (max(ys) + 1) / gh)
            if area(b) >= .015 and b[2] - b[0] >= .05 and b[3] - b[1] >= .04:
                boxes.append(b)
    boxes.sort(key=area, reverse=True)
    return [b for b in boxes[:8] if not any(inside(b, t, .6) for t in table_boxes)]


def caption(box, blks):
    """a 'Figure 3: ...' style line just above or below the box"""
    for b in blks:
        t = b["text"].strip()
        if not re.match(r"(fig(ure)?|table|chart|map|diagram|exhibit)\.?\s*\d*", t, re.I):
            continue
        hx = min(box[2], b["box"][2]) - max(box[0], b["box"][0])
        gap = min(abs(b["box"][1] - box[3]), abs(box[1] - b["box"][3]))
        if hx > 0 and gap < .08:
            return t[:300]
    return ""


def open_pdf(path):
    return pymupdf.open(path)


_an, _body = {}, {}


def _stamp(path):
    from pathlib import Path
    st = Path(path).stat()
    return (str(path), st.st_size, st.st_mtime_ns)


ANALYSIS_VERSION = 3   # 3: labels skip the legend


def analyze(path, pno):
    """blocks + tables + figures + labels for one page. slow the first time (table finding), so it's kept in memory
    and on disk in data/_analysis/ keyed by file hash: a restart doesn't pay it again"""
    k = _stamp(path) + (pno,)
    if k in _an:
        return _an[k]
    from .config import DATA
    from .kb import file_hash
    f = DATA / "_analysis" / f"{file_hash(path)}_p{pno}_v{ANALYSIS_VERSION}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        d["blocks"] = [{**b, "box": tuple(b["box"])} for b in d["blocks"]]
        d["tables"] = [(tuple(b), md) for b, md in d["tables"]]
        d["figures"] = [tuple(b) for b in d["figures"]]
        d["legend"] = tuple(d["legend"]) if d.get("legend") else None
    else:
        with pymupdf.open(path) as pdf:
            p = pdf[pno - 1]
            tabs = tables(p)
            leg = legend_box(p)
            d = {"blocks": blocks(p), "tables": tabs, "figures": figures(p, [t for t, _ in tabs]), "labels": labels(p, leg), "legend": leg}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d), encoding="utf-8")
    _an[k] = d
    return d


def doc_body_size(path):
    k = _stamp(path)
    if k not in _body:
        with pymupdf.open(path) as pdf:
            _body[k] = body_size(pdf)
    return _body[k]


def legend_box(page):
    """box of a map legend / key, or None. found from a 'Legend' or 'Key' heading plus the text packed under it.
    legend entries name things that are also on the map (PARKING RAMP AND LOT), so their positions would mislead"""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip():
                    spans.append((frac(s["bbox"], page), s["text"].strip()))
    heads = [b for b, t in spans if re.fullmatch(r"(map )?(legend|key)", t, re.I)]
    if not heads:
        return None
    h = heads[0]
    below = sorted((b for b, _ in spans if b[1] > h[1] and b[0] >= h[0] - .02), key=lambda b: b[1])
    bottom, right, gap = h[3], h[2], .04   # the first entry can sit a bit under the heading, after that entries are packed
    for b in below:
        if b[1] - bottom > gap:
            break
        bottom, right, gap = max(bottom, b[3]), max(right, b[2]), .015
    return (round(h[0] - .01, 4), round(h[1] - .005, 4), round(min(1, right + .005), 4), round(bottom + .005, 4))


def in_box(x, y, b):
    return b is not None and b[0] <= x <= b[2] and b[1] <= y <= b[3]


def labels(page, legend=None):
    """every short text label with its center on the page, 0..1. this is how the model gets exact positions.
    lines stacked right on top of each other (STUDENT / CENTER / (SC)) become one label. legend entries are skipped"""
    lines = {}
    for x0, y0, x1, y1, w, b, ln, _ in page.get_text("words"):
        lines.setdefault((b, ln), []).append((x0, y0, x1, y1, w))
    boxes = []
    for ws in lines.values():
        boxes.append([min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws), " ".join(w[4] for w in ws).strip()])
    boxes.sort(key=lambda b: (b[1], b[0]))
    groups = []
    for b in boxes:
        h = b[3] - b[1]
        for g in groups:
            last = g[-1]
            ov = min(b[2], last[2]) - max(b[0], last[0])
            if ov > .5 * min(b[2] - b[0], last[2] - last[0]) and 0 <= b[1] - last[3] < 1.2 * h and len(g) < 4:
                g.append(b)
                break
        else:
            groups.append([b])
    W, H = page.rect.width, page.rect.height
    out = []
    for g in groups:
        t = " ".join(b[4] for b in g).strip()
        if len(t) < 2 or len(t) > 80 or re.fullmatch(r"[\d\W]+", t):
            continue
        x0, y0, x1, y1 = min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g)
        if in_box((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, legend):
            continue
        out.append({"text": t[:60], "x": round((x0 + x1) / 2 / W, 3), "y": round((y0 + y1) / 2 / H, 3)})
    return out
