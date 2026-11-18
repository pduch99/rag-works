# FP8: hybrid retrieval + reranking

## what changed

- retrieval: `vector` | `bm25` | `hybrid` (rrf fusion)
- rerank: `none` | `cross` (local cross-encoder) | `gemini` (model orders the candidates, crops included)
- candidate pool setting: retrieve 10-50, rerank, keep top k, send the best few
- trace: every hit shows its vector rank, bm25 rank and how far the reranker moved it

## theory in one line each

- bm25: keyword scoring with term frequency saturation + rare-word weighting. exact where embeddings are fuzzy (ICS-499, 60-89 credits, names).
- reciprocal rank fusion (cormack et al, 2009): score = Σ 1/(60 + rank). uses ranks only, so a cosine distance and a bm25 score never get compared directly.
- bi-encoder vs cross-encoder: embeddings encode question and chunk separately (fast, rough). a cross-encoder reads them together (slow, precise), so use it only on a short candidate list.
- retrieve many, send few: recall comes from the wide pool, precision from the reranker.

## experiment

`python cli.py ladder testbed fp8 --index layout.bge.structured.ctx --split dev`

1. vector
2. bm25
3. hybrid
4. vector + cross rerank
5. hybrid + cross rerank
6. hybrid + gemini rerank

same index the whole way, so only retrieval changes.

## what to watch

- exact_term questions: bm25 / hybrid should win
- mrr and recall@3: reranking should push the right chunk up
- latency: cross rerank adds a little, gemini rerank adds a model call
- whatever still fails with the right evidence retrieved is a generation failure, which is fp10's job
