# text -> vectors. we embed ourselves and hand chroma the numbers, so swapping models is one setting
import os
import threading

from .config import EMBEDDERS

_models, _lock = {}, threading.Lock()
BGE_QUERY = "Represent this sentence for searching relevant passages: "


def _minilm():
    with _lock:
        return _minilm_load()


def _minilm_load():
    if "minilm" not in _models:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _models["minilm"] = DefaultEmbeddingFunction()
    return _models["minilm"]


def _bge():
    with _lock:
        return _bge_load()


def _bge_load():
    if "bge" not in _models:
        from fastembed import TextEmbedding
        _models["bge"] = TextEmbedding(EMBEDDERS["bge"]["model"])   # downloads ~210 mb the first time
    return _models["bge"]


def _gemini(texts, task):
    from .llm import client
    out = []
    for i in range(0, len(texts), 90):
        r = client("gemini").embeddings.create(model=EMBEDDERS["gemini"]["model"], input=texts[i:i + 90])
        out += [d.embedding for d in r.data]
    return out


def embed(embedder, texts, query=False):
    texts = [t if t.strip() else "(empty)" for t in texts]
    if embedder == "minilm":
        return [list(map(float, v)) for v in _minilm()(texts)]
    if embedder == "bge":
        if query:
            texts = [BGE_QUERY + t for t in texts]
        return [v.tolist() for v in _bge().embed(texts)]
    if embedder == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("gemini embeddings need GEMINI_API_KEY in .env")
        return _gemini(texts, "query" if query else "doc")
    raise ValueError(f"unknown embedder {embedder}")
