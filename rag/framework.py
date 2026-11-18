# the framework: rag as 7 stages with switches, and a catalog of rag types described with the same fields.
# every type either maps to switches in this app (implemented) or is described only. the app's framework view
# and docs/framework.md both come from here:  python -m rag.framework  rewrites docs/framework.md

STAGES = [
    ("ingest", "chunk", "how documents become pieces: fixed text, pages, image regions, headings and tables, real figure boxes",
     ["chunker"]),
    ("describe", "describe images", "turn each image crop into searchable text: free prose, or every label plus where it sits",
     ["style"]),
    ("embed", "embed", "text to vectors. which model, and whether each chunk gets a line of page context first",
     ["embedder", "contextual"]),
    ("retrieve", "retrieve", "find candidates: by meaning (vector), by exact words (bm25), or both fused",
     ["retriever"]),
    ("rerank", "rerank", "reorder the candidates with a model that reads question and chunk together",
     ["reranker"]),
    ("evidence", "build evidence", "pick what the model sees and add facts code can work out: label positions, directions",
     ["labels", "spatial"]),
    ("generate", "check + answer", "answer with citations, optionally grade the evidence first and verify the claims after",
     ["check"]),
]

GROUPS = ["indexing", "retrieval", "post-retrieval", "generation control", "architecture"]

# fields every type gets. measured = (step before, step after) from story.py, so the change is isolated
CATALOG = [
    {"id": "naive", "name": "Naive RAG", "group": "indexing", "stages": ["ingest", "embed", "retrieve", "generate"], "status": "implemented",
     "problem": "a model can't answer about documents it never saw",
     "how": "cut text into fixed chunks, embed them, take the top matches by similarity, paste them into the prompt",
     "cost": "1 model call per question. cheapest to build and run",
     "fails": "chunks split facts in half, images are invisible, near-duplicate passages confuse it",
     "when": "plain text documents and simple fact lookups. the baseline everything else is measured against",
     "config": {"chunker": "text", "embedder": "minilm", "style": "prose", "contextual": False, "retriever": "vector", "reranker": "none", "labels": False, "spatial": False, "check": "none"},
     "measured": None, "fp": "fp2"},
    {"id": "multimodal", "name": "Multimodal RAG", "group": "indexing", "stages": ["ingest", "describe"], "status": "implemented",
     "problem": "answers live in maps, charts and diagrams, not in the text layer",
     "how": "crop pages or regions into images, have a vision model describe each crop, index the descriptions, send the crops with the answer",
     "cost": "1 vision call per crop at build time (cached), images make every answer call bigger",
     "fails": "descriptions miss small labels, a crop seam can cut a place in half, the model misreads directions",
     "when": "any corpus with figures that matter: maps, slides, scanned forms, charts",
     "config": {"chunker": "region", "embedder": "minilm", "style": "prose", "contextual": False, "retriever": "vector", "reranker": "none", "labels": False, "spatial": False, "check": "none"},
     "measured": None, "fp": "fp4-fp5"},
    {"id": "structure", "name": "Structure-aware chunking", "group": "indexing", "stages": ["ingest", "describe"], "status": "implemented",
     "problem": "fixed-size chunks ignore the document: a table or a section gets cut mid-way",
     "how": "chunk by headings and sections, keep tables whole as markdown, crop real figure boxes found in the pdf, date lists become one record per line",
     "cost": "a one-time layout pass per page (cached). no extra model calls",
     "fails": "heuristics break on odd layouts (a map's street grid looked like a table, calendar dates looked like headings)",
     "when": "documents with real structure: handbooks, policies, reports, calendars",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": False, "retriever": "vector", "reranker": "none", "labels": False, "spatial": False, "check": "none"},
     "measured": ("fp6 baseline", "fp7 better index"), "fp": "fp7"},
    {"id": "contextual", "name": "Contextual retrieval", "group": "indexing", "stages": ["embed"], "status": "implemented",
     "problem": "a small chunk loses what it's about ('it opens at 8' — what does?)",
     "how": "a model writes one line of page context, which goes in front of every chunk from that page before embedding",
     "cost": "1 text call per page at build time (cached)",
     "fails": "a vague context line adds noise. measured only together with the other fp7 changes here",
     "when": "short chunks from long documents, lots of pronouns or repeated section names",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "vector", "reranker": "none", "labels": False, "spatial": False, "check": "none"},
     "measured": ("fp6 baseline", "fp7 better index"), "fp": "fp7"},
    {"id": "sparse", "name": "Sparse / keyword retrieval", "group": "retrieval", "stages": ["retrieve"], "status": "implemented",
     "problem": "embeddings blur exact terms: codes, names, '60-89 CR'",
     "how": "bm25 scores chunks by how often the question's exact words appear, weighted by how rare they are",
     "cost": "about 1 ms, no model",
     "fails": "misses paraphrases and synonyms, knows nothing about images",
     "when": "ids, course codes, names, legal or technical wording",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "bm25", "reranker": "none", "labels": False, "spatial": False, "check": "none"},
     "measured": None, "fp": "fp8"},
    {"id": "hybrid", "name": "Hybrid RAG", "group": "retrieval", "stages": ["retrieve"], "status": "implemented",
     "problem": "vector search misses exact terms, keyword search misses meaning",
     "how": "run both and merge the rankings with reciprocal rank fusion (k=60): a chunk ranked well by either rises",
     "cost": "about 5 ms more than vector alone, no model calls",
     "fails": "if both rankers are wrong it can't help. fusion can push a good vector-only hit down",
     "when": "almost always worth it. mixed questions, some exact and some fuzzy",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "hybrid", "reranker": "none", "labels": False, "spatial": False, "check": "none"},
     "measured": ("fp7 better index", "fp8 hybrid + rerank"), "fp": "fp8"},
    {"id": "rerank", "name": "Reranked RAG", "group": "post-retrieval", "stages": ["rerank"], "status": "implemented",
     "problem": "first-stage search is fast but rough. the best chunk is often 4th or 9th",
     "how": "take a wider pool, then a cross-encoder reads question + chunk together and reorders (or a model ranks them, crops included)",
     "cost": "cross-encoder about 250 ms on cpu for 10 candidates. the model version costs a call",
     "fails": "text-only rerankers can't judge image crops, and it can only reorder what search found",
     "when": "when recall is fine but the right chunk isn't on top",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "hybrid", "reranker": "cross", "labels": False, "spatial": False, "check": "none"},
     "measured": ("fp7 better index", "fp8 hybrid + rerank"), "fp": "fp8"},
    {"id": "tools", "name": "Evidence-augmented RAG", "group": "post-retrieval", "stages": ["evidence"], "status": "implemented",
     "problem": "the right crop is retrieved but the model still flips north and south",
     "how": "add facts code can compute: exact label positions from the pdf text layer (legend skipped), then directions between the places the question names",
     "cost": "about 1 ms, no model calls",
     "fails": "only works where a text layer exists. a small model can still contradict a fact it was handed",
     "when": "maps, floor plans, diagrams, tables: anything where code can do the math better than the model",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "hybrid", "reranker": "cross", "labels": True, "spatial": True, "check": "auto"},
     "measured": ("fp9 labels + self-check", "fp10 spatial facts"), "fp": "fp9-fp10"},
    {"id": "corrective", "name": "Corrective RAG", "group": "generation control", "stages": ["generate"], "status": "implemented",
     "problem": "the model answers even when the evidence is off-topic, and guesses when nothing was found",
     "how": "a grader marks which evidence is relevant. nothing relevant: search again with a better query, still nothing: say not found",
     "cost": "1 extra call per question, 3 when it retries",
     "fails": "a weak grader drops the right evidence or retries into worse results (seen with the 8b local model)",
     "when": "high stakes answers where 'I don't know' beats a wrong answer",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "hybrid", "reranker": "cross", "labels": True, "spatial": False, "check": "grade"},
     "measured": ("fp8 hybrid + rerank", "fp9 labels + self-check"), "fp": "fp9"},
    {"id": "verify", "name": "Self-verifying RAG", "group": "generation control", "stages": ["generate"], "status": "implemented",
     "problem": "a fluent answer can include claims the evidence doesn't support",
     "how": "after answering, split the answer into claims, check each against the evidence and the crops, rewrite once if any fail. auto = only when confidence is low",
     "cost": "1-2 extra calls, roughly doubles answer time",
     "fails": "a checker built from the same small model shares its blind spots and approves its mistakes",
     "when": "with a strong checker model, or when you need a claim-by-claim audit trail",
     "config": {"chunker": "layout", "embedder": "bge", "style": "structured", "contextual": True, "retriever": "hybrid", "reranker": "cross", "labels": True, "spatial": False, "check": "full"},
     "measured": ("fp8 hybrid + rerank", "fp9 labels + self-check"), "fp": "fp9"},
    {"id": "visual", "name": "Visual document retrieval (ColPali)", "group": "architecture", "stages": ["ingest", "describe", "embed", "retrieve"], "status": "described",
     "problem": "describing images in text loses detail before search even starts",
     "how": "embed page images directly with a vision-language retriever (many vectors per page), search pages by image, no text extraction",
     "cost": "needs a gpu to index, bigger index (many vectors per page)",
     "fails": "returns whole pages, not exact regions. weaker on exact codes and numbers",
     "when": "scanned or image-heavy corpora with no usable text layer",
     "config": None, "measured": None, "fp": "future work (byaldi / colpali-engine)"},
    {"id": "graph", "name": "GraphRAG", "group": "architecture", "stages": ["ingest", "retrieve"], "status": "described",
     "problem": "questions about relationships across many documents ('which offices connect to...')",
     "how": "a model extracts entities and relations into a graph, questions are answered by walking the graph and community summaries",
     "cost": "many model calls to build the graph, slow to update",
     "fails": "extraction errors become graph errors, overkill for single-fact lookups",
     "when": "large connected corpora: org charts, case files, research literature",
     "config": None, "measured": None, "fp": "future work"},
    {"id": "agentic", "name": "Agentic RAG", "group": "architecture", "stages": ["retrieve", "evidence", "generate"], "status": "described",
     "problem": "one search isn't enough for multi-step questions",
     "how": "a model plans, picks tools (search, calculator, the spatial helper), searches several times and decides when it has enough",
     "cost": "many calls per question, latency is unpredictable",
     "fails": "loops, wanders off, hard to evaluate because every run takes a different path",
     "when": "multi-hop questions, when tools exist and latency doesn't matter much",
     "config": None, "measured": None, "fp": "future work. corrective rag's retry is a one-step version"},
]


# the metro state story: each step changes one stage from the one before (same names story.py writes in run notes)
STORY = [
    ("fp6 baseline", "baseline", "fixed text + 2x2 image quadrants, prose descriptions, vector search"),
    ("fp7 better index", "indexing", "structure + layout chunks, bge embeddings, structured descriptions, page context"),
    ("fp8 hybrid + rerank", "retrieval", "vector + bm25 fused, cross-encoder rerank"),
    ("fp9 labels + self-check", "evidence + checks", "exact label positions, grade evidence, verify when unsure"),
    ("fp10 spatial facts", "evidence", "directions between the named places, computed by code"),
]


def classify(cfg):
    """which catalog types this config is using. cfg has chunker, embedder, style, contextual, retriever, reranker, labels, spatial, check"""
    on = []
    if cfg.get("chunker") in ("page", "region", "region3", "structure", "layout"):
        on.append("multimodal")
    if cfg.get("chunker") in ("structure", "layout"):
        on.append("structure")
    if cfg.get("contextual"):
        on.append("contextual")
    if cfg.get("retriever") == "bm25":
        on.append("sparse")
    if cfg.get("retriever") == "hybrid":
        on.append("hybrid")
    if cfg.get("reranker", "none") != "none":
        on.append("rerank")
    if cfg.get("labels") or cfg.get("spatial"):
        on.append("tools")
    if cfg.get("check") in ("grade", "full", "auto"):
        on.append("corrective")
    if cfg.get("check") in ("verify", "full", "auto"):
        on.append("verify")
    return on or ["naive"]


def story_runs(runs, provider):
    """latest story.py run per step for one provider: {step name: run}"""
    out = {}
    for r in runs:   # newest first
        n = r.get("note", "")
        if r.get("provider") == provider and n.startswith("story: ") and n[7:] not in out:
            out[n[7:]] = r
    return out


def measured(t, steps):
    """(before summary, after summary) for a type, or None"""
    if not t["measured"]:
        return None
    a, b = t["measured"]
    if a in steps and b in steps:
        return steps[a]["summary"], steps[b]["summary"]
    return None


def markdown():
    out = ["# The RAG framework", "",
           "RAG as seven stages. Every RAG type changes one or more of them. Each type is described with the same fields, "
           "so they can be compared. Implemented types map to switches in RAG Works and are measured on the same gold questions "
           "(`python story.py`). This file is generated from `rag/framework.py`.", "", "## Stages", "",
           "| # | stage | what it decides | switches in RAG Works |", "|---|---|---|---|"]
    for i, (sid, name, what, sw) in enumerate(STAGES, 1):
        out.append(f"| {i} | {name} | {what} | {', '.join(f'`{s}`' for s in sw)} |")
    for g in GROUPS:
        out += ["", f"## {g.capitalize()}", ""]
        for t in [t for t in CATALOG if t["group"] == g]:
            out += [f"### {t['name']}", "",
                    f"- **status:** {t['status']} ({t['fp']})",
                    f"- **stages:** {', '.join(t['stages'])}",
                    f"- **problem it solves:** {t['problem']}",
                    f"- **how it works:** {t['how']}",
                    f"- **cost:** {t['cost']}",
                    f"- **how it fails:** {t['fails']}",
                    f"- **use it when:** {t['when']}"]
            if t["measured"]:
                out.append(f"- **measured:** `{t['measured'][0]}` → `{t['measured'][1]}` in story.py")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    from .config import ROOT
    p = ROOT / "docs" / "framework.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text(markdown(), encoding="utf-8")
    print(f"wrote {p}")
