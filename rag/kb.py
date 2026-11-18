# knowledge bases. a kb is just a folder: corpus/<name>/docs/*  (+ optional gold.json, kb.json)
import hashlib
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image

from .config import CORPUS, DATA, DOC_TYPES, RENDER_PX

Image.MAX_IMAGE_PIXELS = None


def list_kbs():
    return sorted(p.name for p in CORPUS.iterdir() if (p / "docs").is_dir()) if CORPUS.exists() else []


def kb_dir(kb):
    return CORPUS / kb


def data_dir(kb):
    d = DATA / kb
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_kb(name):
    name = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
    if len(name) < 3:
        raise ValueError("kb name needs at least 3 letters or numbers")
    (CORPUS / name / "docs").mkdir(parents=True, exist_ok=True)
    return name


def docs(kb):
    d = kb_dir(kb) / "docs"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in DOC_TYPES) if d.exists() else []


def save_upload(kb, filename, data):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
    path = kb_dir(kb) / "docs" / name
    path.write_bytes(data)
    return path


def info(kb):
    p = kb_dir(kb) / "kb.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def gold(kb):
    p = kb_dir(kb) / "gold.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


_hash, _pages = {}, {}


def stamp(path):
    st = Path(path).stat()
    return (str(path), st.st_size, st.st_mtime_ns)


def file_hash(path):
    """sha1 of the file, remembered until the file changes (no re-reading 13 mb pdfs every click)"""
    k = stamp(path)
    if k not in _hash:
        _hash[k] = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    return _hash[k]


def kind(path):
    s = path.suffix.lower()
    return "pdf" if s == ".pdf" else "text" if s in {".txt", ".md"} else "image"


def has_visual(page, text):
    # skip describing plain text pages. saves a lot of vision calls on big pdfs
    return bool(page.get_images()) or len(page.get_drawings()) > 20 or len(text.strip()) < 200


def pages(kb, path):
    """one dict per page: doc, page, text, image (rendered png or None), visual (worth describing?). remembered until the file changes"""
    key = (kb,) + stamp(path)
    if key in _pages and any(p["image"] and not p["image"].exists() for p in _pages[key]):
        del _pages[key]   # data/ was deleted, render again
    if key not in _pages:
        _pages[key] = _pages_uncached(kb, path)
    return [dict(p) for p in _pages[key]]


def _pages_uncached(kb, path):
    k = kind(path)
    if k == "text":
        return [{"doc": path.name, "page": 1, "text": path.read_text(encoding="utf-8", errors="ignore"), "image": None, "visual": False}]
    out_dir = data_dir(kb) / "pages"
    out_dir.mkdir(exist_ok=True)
    if k == "image":
        img = Image.open(path).convert("RGB")
        img.thumbnail((RENDER_PX, RENDER_PX))
        out = out_dir / f"{path.name}_p1.png"
        img.save(out)
        return [{"doc": path.name, "page": 1, "text": "", "image": out, "visual": True}]
    result = []
    pdf = pymupdf.open(path)
    for i, page in enumerate(pdf, start=1):
        text = page.get_text()
        out = out_dir / f"{path.name}_p{i}.png"
        if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
            zoom = RENDER_PX / max(page.rect.width, page.rect.height)
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(out)
        result.append({"doc": path.name, "page": i, "text": text, "image": out, "visual": has_visual(page, text)})
    pdf.close()
    return result


_counts = {}


def page_count(path):
    if kind(path) != "pdf":
        return 1
    k = stamp(path)
    if k not in _counts:
        with pymupdf.open(path) as pdf:
            _counts[k] = len(pdf)
    return _counts[k]


def thumb(kb, path, px=160):
    """small jpg of page 1 for lists. text files get None"""
    k = kind(path)
    if k == "text":
        return None
    out_dir = data_dir(kb) / "thumbs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{path.name}.jpg"
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        if k == "pdf":
            with pymupdf.open(path) as pdf:
                p = pdf[0]
                zoom = px * 2 / max(p.rect.width, p.rect.height)
                pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            img = Image.open(path).convert("RGB")
        img.thumbnail((px * 2, px * 2))
        img.save(out, "JPEG", quality=80)
    return out
