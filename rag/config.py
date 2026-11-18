# settings in one place. anything here can be overridden in .env
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CORPUS = ROOT / "corpus"   # your docs + gold questions + description cache (commit this)
DATA = ROOT / "data"       # generated: chroma, page renders, crops (safe to delete)
RUNS = ROOT / "runs"       # eval results (commit these, they're your evidence)

DOC_TYPES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md"}

# fixed text chunking (fp2 baseline). chunks never cross a page
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# page render size. longest side in px
RENDER_PX = int(os.getenv("RENDER_PX", 3000))
# images sent to a model get shrunk to this
SEND_PX = int(os.getenv("SEND_PX", 1600))

# chunkers = how a doc becomes chunks. text = how text is cut, visual = how page images are cut
CHUNKERS = {
    "text": {"text": "fixed", "visual": None, "label": "text only", "about": "fixed 1000/200 text chunks, no images (fp2 / fp4 text-only baseline)"},
    "page": {"text": "fixed", "visual": "full", "label": "page-level", "about": "fixed text chunks + one visual chunk per page (fp4)"},
    "region": {"text": "fixed", "visual": "2x2", "label": "region 2x2", "about": "fixed text chunks + each page cut into nw/ne/sw/se (fp5)"},
    "region3": {"text": "fixed", "visual": "3x3", "label": "region 3x3", "about": "fixed text chunks + each page cut into a 3x3 grid"},
    "structure": {"text": "structure", "visual": "2x2", "label": "structure + 2x2", "about": "headings / sections / tables as chunks + 2x2 regions (fp7, isolates the text change)"},
    "layout": {"text": "structure", "visual": "layout", "label": "structure + layout", "about": "structure text + real figure / table boxes from the pdf, big figures split 2x2 (fp7)"},
}
STRATEGIES = CHUNKERS   # old name

# embedding models. all run through our own code, chroma just stores vectors
EMBEDDERS = {
    "minilm": {"label": "minilm (fp6 default)", "about": "chroma default all-MiniLM-L6-v2. only reads ~256 tokens"},
    "bge": {"label": "bge-base (local)", "about": "BAAI/bge-base-en-v1.5 via fastembed, 512 tokens, runs on cpu", "model": os.getenv("BGE_MODEL", "BAAI/bge-base-en-v1.5")},
    "gemini": {"label": "gemini embedding", "about": "google embedding api, long input, uses free-tier quota", "model": os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")},
}
DEFAULT_EMBEDDER = os.getenv("RAG_EMBEDDER", "bge")

# how images get described for search
DESCRIBE_STYLES = {"prose": "free text description (fp4 / fp5)", "structured": "every label + where it sits, then a summary (fp7)"}

RETRIEVERS = {
    "vector": "embedding similarity only (fp2-fp7)",
    "bm25": "keyword match only (exact terms, codes, names)",
    "hybrid": "vector + bm25 fused with reciprocal rank fusion (fp8)",
}
RERANKERS = {
    "none": "keep the retriever's order",
    "cross": "local cross-encoder reads question + chunk together (fp8)",
    "gemini": "the model looks at the candidates, crops included, and orders them (fp8)",
}
CHECKS = {
    "none": "answer straight from the retrieved evidence (fp2-fp8)",
    "grade": "grade the evidence first: drop what's irrelevant, retry once with a better query, say 'not found' if nothing helps",
    "verify": "check the answer claim by claim against the evidence (crops included), fix it once if something isn't backed",
    "full": "grade + verify",
    "auto": "grade always, verify only when confidence is low (cheaper full)",
}
CROSS_MODEL = os.getenv("CROSS_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RRF_K = 60

# providers. groq and gemini have free tiers, ollama runs on your own gpu
PROVIDERS = {
    "groq": {
        "key": "GROQ_API_KEY",
        "text": os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-20b"),
        "vision": os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b"),
        "max_images": 3,          # groq vision cap per request
        "image_tokens": 2048,     # groq bills every image as 2048 input tokens
        "input_budget": 6900,     # groq rejects >7000 input tokens per request on the free tier
    },
    "gemini": {
        "key": "GEMINI_API_KEY",
        "text": os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash"),
        "vision": os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
        "max_images": 8,
        "image_tokens": 1300,
        "input_budget": 60000,
    },
    "ollama": {
        "key": None,              # local, no key and no rate limits. needs the ollama app running
        "text": os.getenv("OLLAMA_TEXT_MODEL", "qwen3-vl:8b-instruct"),
        "vision": os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b-instruct"),   # same model for both so the gpu never swaps
        "max_images": 2,
        "image_tokens": 1300,
        "input_budget": 6000,     # fits inside an 8192 context with room for the answer
    },
}
DEFAULT_PROVIDER = os.getenv("RAG_PROVIDER", "gemini")


def has_key(provider):
    k = PROVIDERS[provider]["key"]
    return k is None or bool(os.getenv(k))


def ix_key(chunker, embedder="minilm", style="prose", contextual=False):
    """name of one index build, e.g. layout.bge.structured.ctx"""
    return f"{chunker}.{embedder}.{style}" + (".ctx" if contextual else "")


def parse_ix(key):
    p = key.split(".")
    if len(p) == 1:   # fp6 index names were just the chunker
        return {"chunker": p[0], "embedder": "minilm", "style": "prose", "contextual": False}
    return {"chunker": p[0], "embedder": p[1], "style": p[2], "contextual": len(p) > 3 and p[3] == "ctx"}


DESCRIBE_PROMPT = """You are indexing one {what} from page {page} of "{doc}" for a search engine.
Describe only what is clearly visible in this image. Include:
- every readable label, heading and name, written exactly as shown
- what kind of thing it is (map, chart, table, diagram, photo, form, text block)
- where things are relative to each other (left/right/above/below, north/south if a compass is shown)
- numbers, legends and anything a person might ask a question about
{hint}
Plain text, under 250 words. Do not guess about anything you cannot read."""

DESCRIBE_STRUCTURED = """You are indexing one {what} from page {page} of "{doc}" for a search engine.
Write exactly two parts.
LABELS:
every readable label, heading, name and number in the image, one per line, as "label | where", where `where` is its spot
inside this image as horizontal,vertical percent from the top-left (example: "PARKING RAMP | 62,40"). Copy text exactly.
SUMMARY:
what this image is (map, chart, table, diagram, photo, form) and how the labeled things relate to each other
(left/right/above/below, north/south if a compass is shown), plus key numbers, trends or steps. Under 150 words.
{hint}
Do not guess about anything you cannot read."""

CONTEXT_PROMPT = """Here is page {page} of the document "{doc}".
<page>
{text}
</page>
In one or two sentences, say what this document is and what this page covers (section, topic), so that a short
chunk cut from this page can be found by a search engine. Plain text only, no preamble."""