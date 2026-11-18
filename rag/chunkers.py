# docs -> chunks. text chunks (fixed or structure) + visual chunks (page, grid or layout) + optional context prefix
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from . import kb as kbs
from . import layout as lay
from .config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKERS, CONTEXT_PROMPT, CORPUS, DESCRIBE_PROMPT, DESCRIBE_STRUCTURED, PROVIDERS, ROOT
from .llm import chat

Image.MAX_IMAGE_PIXELS = None
LLM_WORKERS = int(os.getenv("LLM_WORKERS", 3))   # parallel model calls during builds. lower it if you hit rate limits
_lock = threading.Lock()
CROP_OVERLAP = float(os.getenv("CROP_OVERLAP", .15))   # split figures overlap this much past the middle, so a label on the seam is whole in one crop
CHUNK_VERSION = 3
RECORD_CHUNK = int(os.getenv("RECORD_CHUNK", 500))   # date lists get smaller chunks: each one is about fewer events, so search is sharper   # bump when chunking changes so old indexes show as stale and rebuild

# region boxes as fractions of the page (x0, y0, x1, y1). up = north unless the doc says otherwise
GRIDS = {
    "full": [("full", (0, 0, 1, 1))],
    "2x2": [("nw", (0, 0, .5, .5)), ("ne", (.5, 0, 1, .5)), ("sw", (0, .5, .5, 1)), ("se", (.5, .5, 1, 1))],
    "3x3": [(n, (c / 3, r / 3, (c + 1) / 3, (r + 1) / 3)) for r, row in enumerate([["nw", "n", "ne"], ["w", "c", "e"], ["sw", "s", "se"]]) for c, n in enumerate(row)],
}
WHERE = {"full": "whole page", "nw": "top-left", "n": "top-middle", "ne": "top-right", "w": "middle-left", "c": "center",
         "e": "middle-right", "sw": "bottom-left", "s": "bottom-middle", "se": "bottom-right"}


def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/") if p else ""


def boxstr(b):
    return ",".join(f"{x:.4f}" for x in b) if b else ""


def where_on_page(b):
    """plain words for where a box sits, e.g. 'top-left'"""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    v = "top" if cy < .34 else "bottom" if cy > .66 else "middle"
    hz = "left" if cx < .34 else "right" if cx > .66 else "center"
    return "center" if v == "middle" and hz == "center" else f"{v}-{hz}"


MIN_SECTION = 350


def fixed_chunks(text):
    text = text.strip()
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + CHUNK_SIZE])
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return out


def text_meta(page, i, box=None, sub="fixed", heading=""):
    return {"doc": page["doc"], "page": page["page"], "page_image": rel(page["image"]), "kind": "text", "sub": sub, "chunk": i,
            "region": "", "grid": "", "box": boxstr(box), "crop": "", "label": f"{sub} chunk {i}" if sub != "table" else f"table {i}",
            "where": "", "heading": heading[:120]}


def structure_pdf(path, pages, fig_boxes):
    """chunks follow headings. tables become markdown chunks. a heading carries over to the next page"""
    body = lay.doc_body_size(path)
    heading, out = "", []
    for page in pages:
        an = lay.analyze(path, page["page"])
        tabs = an["tables"]
        figs = [b for b in an["figures"] if lay.area(b) < .5]   # labels inside small figures belong to the figure
        buf, n = [], 0

        def flush():
            nonlocal buf, n
            if buf:
                body_text = "\n".join(b["text"] for b in buf)
                text = (heading + "\n" + body_text) if heading and not body_text.startswith(heading) else body_text
                out.append({"text": text, "meta": text_meta(page, n, lay.union([b["box"] for b in buf]), "section", heading)})
                n += 1
                buf = []
        for b in an["blocks"]:
            if any(lay.inside(b["box"], t) for t, _ in tabs) or any(lay.inside(b["box"], f, .8) for f in figs):
                continue
            if lay.is_heading(b, body):
                if sum(len(x["text"]) for x in buf) >= MIN_SECTION:   # tiny sections ride along with the next one
                    flush()
                    heading = b["text"].replace("\n", " ").strip()
                elif not buf:
                    heading = b["text"].replace("\n", " ").strip()
                buf.append(b)
                continue
            if buf and sum(len(x["text"]) for x in buf) + len(b["text"]) > CHUNK_SIZE:
                flush()
            buf.append(b)
        flush()
        for t, md in tabs:
            out.append({"text": (heading + "\n" if heading else "") + md, "meta": text_meta(page, n, t, "table", heading)})
            n += 1
    return out


DATE_LINE = re.compile(r"^(mon|tues|wednes|thurs|fri|satur|sun)day,?\s+[a-z]+\.?\s+\d{1,2},?\s+\d{4}$", re.I)


def date_records(lines):
    """calendars list 'date' on one line and 'what happens' on the next. glue them into one line so a chunk
    can never split a date from its event, and a date never gets mistaken for a heading"""
    if sum(bool(DATE_LINE.match(l.strip())) for l in lines) < 5:
        return lines, False
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if DATE_LINE.match(s) and i + 1 < len(lines) and lines[i + 1].strip() and not DATE_LINE.match(lines[i + 1].strip()):
            out.append(f"{s}: {lines[i + 1].strip()}")
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return out, True


def structure_text(page):
    """txt / md: markdown headings or short title-like lines start a section, paragraphs fill it.
    date lists (calendars) become one record per line, and the records start their own section"""
    lines, dated = date_records(page["text"].splitlines())
    first_rec = next((i for i, l in enumerate(lines) if dated and ": " in l and DATE_LINE.match(l.split(": ")[0].strip())), None)
    heading, buf, out = "", [], []

    def flush():
        nonlocal buf
        body = "\n".join(buf).strip()
        if body:
            out.append({"text": (heading + "\n" + body) if heading else body, "meta": text_meta(page, len(out), None, "section", heading)})
        buf = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i == first_rec:   # page header / navigation stays its own chunk, records start fresh under the last short line
            flush()
            heading = next((x.strip() for x in reversed(lines[:i]) if x.strip() and len(x.strip()) < 40), heading)
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        rec = dated and ": " in s and bool(DATE_LINE.match(s.split(": ")[0].strip()))
        title_like = s and not rec and len(s) < 70 and not s.endswith((".", ",", ":", ";")) and len(nxt) > len(s) + 20
        if s.startswith("#") or title_like:
            if sum(len(x) for x in buf) >= MIN_SECTION:   # tiny sections ride along with the next one
                flush()
                heading = s.lstrip("# ").strip()
                continue
            if not buf:
                heading = s.lstrip("# ").strip()
                continue
        if sum(len(x) for x in buf) + len(s) > (RECORD_CHUNK if rec else CHUNK_SIZE):
            flush()
        buf.append(line)
    flush()
    return out


def visual_plan(path, page, grid):
    """[(region name, box, where words, label)] for one page"""
    if grid in ("full", "2x2", "3x3"):
        return [(r, b, WHERE[r] if r != "full" else "whole page", r.upper() if r != "full" else "page image") for r, b in GRIDS[grid]]
    # layout: real figures from the pdf. a figure filling most of the page gets split 2x2 so maps stay precise
    if kbs.kind(path) != "pdf":
        figs = [(0, 0, 1, 1)]
    else:
        figs = lay.analyze(path, page["page"])["figures"]
    plan = []
    for i, f in enumerate(figs, start=1):
        pad = (max(0, f[0] - .01), max(0, f[1] - .01), min(1, f[2] + .01), min(1, f[3] + .01))
        if lay.area(pad) > .45:
            for r, b in GRIDS["2x2"]:
                o = CROP_OVERLAP
                b = (max(0, b[0] - o) if b[0] else 0, max(0, b[1] - o) if b[1] else 0, min(1, b[2] + o) if b[2] < 1 else 1, min(1, b[3] + o) if b[3] < 1 else 1)
                sub = (pad[0] + b[0] * (pad[2] - pad[0]), pad[1] + b[1] * (pad[3] - pad[1]), pad[0] + b[2] * (pad[2] - pad[0]), pad[1] + b[3] * (pad[3] - pad[1]))
                plan.append((f"f{i}{r}", sub, f"{WHERE[r]} part of figure {i}", f"FIG{i} {r.upper()}"))
        else:
            plan.append((f"f{i}", pad, f"figure {i}, {where_on_page(pad)} of the page", f"FIG{i}"))
    if not plan and len(page["text"].strip()) < 200:
        plan = [("full", (0, 0, 1, 1), "whole page", "page image")]   # scanned or image-only page
    return plan


def crop(kb, page, tag, region, box):
    if box == (0, 0, 1, 1):
        return page["image"]
    out_dir = kbs.data_dir(kb) / "crops"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{page['image'].stem}_{tag}_{region}.png"
    if not out.exists() or out.stat().st_mtime < page["image"].stat().st_mtime:
        img = Image.open(page["image"])
        w, h = img.size
        img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).save(out)
    return out


POS = re.compile(r"^(\s*[-*]?\s*)(.+?)\s*\|\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$")


def to_page(desc, box, legend=None):
    """structured descriptions give positions inside the crop. turn them into whole-page positions like the label list,
    so 'north' means the same thing in every crop. labels inside the legend get marked as legend instead"""
    if not box:
        return desc
    out = []
    for line in desc.splitlines():
        m = POS.match(line)
        if not m:
            out.append("LABELS (positions on the whole page, 0-100, x grows east, y grows south):" if line.strip() == "LABELS:" else line)
            continue
        x = box[0] + int(m.group(3)) / 100 * (box[2] - box[0])
        y = box[1] + int(m.group(4)) / 100 * (box[3] - box[1])
        out.append(f"{m.group(1)}{m.group(2)} | " + ("legend" if lay.in_box(x, y, legend) else f"{x * 100:.0f},{y * 100:.0f}"))
    return "\n".join(out)


def cache_file(kb, name):
    p = CORPUS / kb / "cache" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def describe(kb, cache, provider, fhash, page, grid, region, box, where, image, style, log=print):
    """vision description of one crop. cached forever per file hash + spot + style, so we only pay once"""
    spot = f"{grid}:{region}" if grid != "layout" else "layout:" + boxstr(box)
    key = f"{fhash}|p{page['page']}|{spot}" + ("|s" if style == "structured" else "")
    if key in cache:
        return cache[key]["text"]
    log(f"describing {page['doc']} p{page['page']} {region} (new vision call)")
    hint = kbs.info(kb).get("describe_hint", "")
    tmpl = DESCRIBE_STRUCTURED if style == "structured" else DESCRIBE_PROMPT
    prompt = tmpl.format(what="full page" if where == "whole page" else f"crop ({where})", page=page["page"], doc=page["doc"], hint=f"Also: {hint}" if hint else "")
    model = PROVIDERS[provider]["vision"]
    text, _ = chat(provider, model, [prompt, image], max_tokens=700)
    with _lock:
        cache[key] = {"doc": page["doc"], "page": page["page"], "grid": grid, "region": region, "style": style, "text": text, "model": model, "at": time.strftime("%Y-%m-%d %H:%M")}
        cache_file(kb, "descriptions.json").write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return text


def page_context(kb, cache, provider, fhash, page):
    """contextual chunking: one short 'what is this page' line per page, put in front of every chunk from it"""
    key = f"{fhash}|p{page['page']}"
    if key in cache:
        return cache[key]["text"]
    text = page["text"].strip()[:6000] or "(no text layer, image only)"
    out, _ = chat(provider, PROVIDERS[provider]["text"], [CONTEXT_PROMPT.format(page=page["page"], doc=page["doc"], text=text)], max_tokens=300)
    out = re.sub(r"\s+", " ", out).strip()[:400]
    with _lock:
        cache[key] = {"doc": page["doc"], "page": page["page"], "text": out, "at": time.strftime("%Y-%m-%d %H:%M")}
        cache_file(kb, "context.json").write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return out


def build_chunks(kb, chunker, provider, style="prose", contextual=False, log=print):
    """vision calls (descriptions, page contexts) run a few at a time in parallel, so a build waits on the api, not on us"""
    cfg = CHUNKERS[chunker]
    dcache, ccache = load_json(cache_file(kb, "descriptions.json")), load_json(cache_file(kb, "context.json"))
    chunks, jobs, ctx_jobs = [], [], []
    for path in kbs.docs(kb):
        fhash = kbs.file_hash(path)
        pages = kbs.pages(kb, path)
        is_pdf = kbs.kind(path) == "pdf"
        doc_chunks = []
        # text
        if cfg["text"] == "structure" and is_pdf:
            doc_chunks += structure_pdf(path, pages, None)
        elif cfg["text"] == "structure" and kbs.kind(path) == "text":
            doc_chunks += structure_text(pages[0])
        else:
            for pg in pages:
                doc_chunks += [{"text": t, "meta": text_meta(pg, i)} for i, t in enumerate(fixed_chunks(pg["text"]))]
        # visuals: plan them now, describe them all at once below
        if cfg["visual"]:
            tag = cfg["visual"]
            for pg in pages:
                if not pg["image"] or (tag != "layout" and not pg["visual"]):
                    continue
                blks = lay.analyze(path, pg["page"])["blocks"] if is_pdf and tag == "layout" else []
                for region, box, where, label in visual_plan(path, pg, tag):
                    img = crop(kb, pg, tag, region, box)
                    grid = tag if tag != "layout" else ("full" if region == "full" else "layout")
                    cap = lay.caption(box, blks) if blks else ""
                    leg = lay.analyze(path, pg["page"]).get("legend") if is_pdf else None
                    c = {"text": "", "cap": cap, "_pos": (box, leg) if style == "structured" else None, "meta": {
                        "doc": pg["doc"], "page": pg["page"], "page_image": rel(pg["image"]), "kind": "visual", "sub": tag, "chunk": 0,
                        "region": region, "grid": grid, "box": boxstr(box), "crop": rel(img), "label": label, "where": where, "heading": ""}}
                    doc_chunks.append(c)
                    jobs.append((c, (kb, dcache, provider, fhash, pg, grid, region, box, where, img, style, log)))
        if contextual:
            ctx_jobs += [(pg, doc_chunks, (kb, ccache, provider, fhash, pg)) for pg in pages]
        chunks += doc_chunks
    with ThreadPoolExecutor(LLM_WORKERS) as pool:
        for (c, _), desc in zip(jobs, pool.map(lambda j: describe(*j[1]), jobs)):
            if c["_pos"]:
                desc = to_page(desc, *c["_pos"])
            c["text"] = (f"Caption: {c['cap']}\n" if c["cap"] else "") + desc
        for (pg, doc_chunks, _), ctx in zip(ctx_jobs, pool.map(lambda j: page_context(*j[2]), ctx_jobs)):
            for c in doc_chunks:
                if c["meta"]["page"] == pg["page"] and c["meta"]["doc"] == pg["doc"]:
                    c["meta"]["context"] = ctx
    for c in chunks:
        c.pop("cap", None)
        c.pop("_pos", None)
        m = c["meta"]
        c["id"] = f"{m['doc']}|p{m['page']}|" + (f"{m['grid']}:{m['region']}" if m["kind"] == "visual" else f"{m['sub']}{m['chunk']}")
        c["embed_text"] = (f"[{m['doc']} p{m['page']}] {m['context']}\n" if m.get("context") else "") + c["text"]
    seen, out = set(), []
    for c in chunks:   # ids must be unique for chroma
        base, i = c["id"], 1
        while c["id"] in seen:
            c["id"] = f"{base}#{i}"; i += 1
        seen.add(c["id"])
        out.append(c)
    return out
