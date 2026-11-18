# FP7: test bed + better index

## what changed

| fp6 | fp7 |
|---|---|
| 13 hand-written questions, map regions as evidence | gold v2: evidence is {doc, page, box, quote}, key facts, answerable, dev/test split, a test bed editor + ai drafting |
| retrieval pass / region recall | + recall@1/3/5, mrr, key-fact match, results by question type, judge-vs-you agreement |
| minilm embeddings (reads ~256 tokens, cut our region descriptions in half) | + bge-base (512 tokens, local) and gemini embeddings |
| prose image descriptions | + structured: every label with its position, then a summary |
| fixed 1000/200 text chunks | + structure chunks: headings, sections, tables as markdown |
| blind 2x2 / 3x3 crops | + layout chunks: real figure / table boxes from the pdf, captions attached, big figures (maps) split 2x2 |
| chunk text only | + contextual chunks: a "what is this page" line in front of every chunk before embedding |

## theory in one line each

- embedding truncation: a model that only reads 256 tokens never sees the second half of a long chunk.
- structure-aware chunking: cut where the author cut (headings, tables), not every 1000 characters.
- layout-aware visual chunks: a figure is the natural unit. quadrants slice charts and tables in half.
- contextual retrieval (anthropic, 2024): a chunk saying "it costs $50" is unfindable without "what is it". prepend context before embedding.
- evidence by location, not chunk id: lets different chunkers be compared fairly.

## experiment

`python cli.py ladder testbed fp7 --split dev`, one change per step:

1. fp6 baseline: region.minilm.prose
2. + bge embedder
3. + structured descriptions
4. + structure text chunks
5. + layout figure chunks
6. + contextual chunks

retrieval stays plain vector in all of them, so only the index changes.

## what to show

- chunks tab: the chunk boxes drawn on a page (teal text, amber table, plum figure)
- compare tab: the 6 runs, recall@5 / mrr / answer, and the by-type heatmap
- one question that moved from fail to pass, with the before/after trace
- the fp8 setup: what still fails (exact terms, right chunk ranked too low)
