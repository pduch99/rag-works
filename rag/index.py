# chunks -> chroma. one collection per kb + index config (chunker.embedder.style[.ctx])
import json
import time

import chromadb

from . import kb as kbs
from .chunkers import CHUNK_VERSION, GRIDS, WHERE, build_chunks   # noqa: F401  (re-exported for the app)
from .config import CHUNK_OVERLAP, CHUNK_SIZE, ix_key, parse_ix   # noqa: F401
from .embed import embed

_clients, _chunk_cache = {}, {}


def db(kb):
    path = str(kbs.data_dir(kb) / "chroma")
    if path not in _clients:
        _clients[path] = chromadb.PersistentClient(path=path)
    return _clients[path]


def collection(kb, key, fresh=False):
    c = db(kb)
    if fresh:
        try:
            c.delete_collection(key)
        except Exception:
            pass
    return c.get_or_create_collection(name=key, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)


def build(kb, chunker, provider, embedder="minilm", style="prose", contextual=False, log=print):
    key = ix_key(chunker, embedder, style, contextual)
    t0 = time.time()
    chunks = build_chunks(kb, chunker, provider, style, contextual, log)
    log(f"embedding {len(chunks)} chunks with {embedder}")
    vecs = embed(embedder, [c["embed_text"] for c in chunks]) if chunks else []
    col = collection(kb, key, fresh=True)
    for i in range(0, len(chunks), 100):
        part = chunks[i:i + 100]
        col.add(ids=[c["id"] for c in part], embeddings=vecs[i:i + 100], documents=[c["embed_text"] for c in part],
                metadatas=[{k: v for k, v in c["meta"].items() if v is not None} for c in part])
    stamp = {"key": key, "chunker": chunker, "embedder": embedder, "style": style, "contextual": contextual, "provider": provider,
             "chunks": len(chunks), "visual": sum(c["meta"]["kind"] == "visual" for c in chunks), "seconds": round(time.time() - t0, 1),
             "built": time.strftime("%Y-%m-%d %H:%M:%S"), "version": CHUNK_VERSION, "docs": {p.name: kbs.file_hash(p) for p in kbs.docs(kb)}}
    (kbs.data_dir(kb) / f"index_{key}.json").write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    _chunk_cache.pop((kb, key), None)
    log(f"indexed {len(chunks)} chunks into {kb}/{key}")
    return stamp


def status(kb, key):
    """index info, or None if not built. 'stale' = docs changed since build"""
    p = kbs.data_dir(kb) / f"index_{key}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    s["stale"] = s["docs"] != {d.name: kbs.file_hash(d) for d in kbs.docs(kb)} or s.get("version", 1) != CHUNK_VERSION
    return s


def built(kb):
    return sorted(p.stem[6:] for p in kbs.data_dir(kb).glob("index_*.json"))


def all_chunks(kb, key):
    """every chunk of an index. cached until the index is rebuilt"""
    st = status(kb, key)
    tag = st["built"] if st else None
    hit = _chunk_cache.get((kb, key))
    if hit and hit[0] == tag:
        return hit[1]
    r = collection(kb, key).get(include=["documents", "metadatas"])
    out = [{"id": i, "text": d, "meta": m} for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]
    _chunk_cache[(kb, key)] = (tag, out)
    return out


def box_of(meta):
    return tuple(float(x) for x in meta["box"].split(",")) if meta.get("box") else None
