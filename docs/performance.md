# performance: where the time goes

goal: every second a user waits should be the model api, not our code. every answer now shows the split
("our code 14 ms · model api 2.1s") and the trace draws it as a bar. `python cli.py bench <kb> <chunker> ...` times
our side alone with zero api calls.

## measured (this machine, before -> after fp9 work)

| what | before | after | how |
|---|---|---|---|
| app rerun (any click) | ~250 ms | ~60-70 ms | bottom section only runs the view you're on (st.tabs ran all 5 every click) |
| page overlay drawing | ~120 ms every rerun | once, then ~0 | cached by page + boxes |
| showing crops / pages | re-read + re-send multi-mb pngs every rerun | small cached jpegs | `pv()` previews |
| hashing docs (index status) | re-read every pdf every rerun (13 mb map ~10-25 ms) | once per file version | memo keyed on size + mtime |
| page analysis (tables, figures, labels) | ~3 s for the map, 3x per build | once ever, kept in data/_analysis | memo + disk cache by file hash |
| encoding images for the model | 40-150 ms per image per question | once per crop | memo keyed on file version |
| first question after start | paid model load (bge, cross-encoder) + page analysis | warmed in a background thread at startup | `warm()` |
| eval run (14 questions, fake 0.3 s api) | 20.6 s | 7.6 s | 3 questions in flight at once (`EVAL_WORKERS`) |
| index build vision calls | one at a time | 3 at a time | `LLM_WORKERS` |
| rate-limit waits | counted as our time | counted as model time | timer starts before retries |

per question, our code (retrieval + fusion + rerank + labels + evidence selection + prompt) measured 8-19 ms with the
test models. with the real local models expect roughly:
- bge-base question embedding: ~15-40 ms on a laptop cpu
- cross-encoder rerank of 20 candidates: ~50-200 ms on cpu (scales with the candidate pool setting)
- everything else: under 20 ms

run `python cli.py bench testbed layout --contextual --retriever hybrid --reranker cross` on your machine for real numbers.

## what's left that isn't the api

- **cold start** (once per app start): python imports (chromadb, pandas, altair) ~2-3 s.
- **first install**: bge-base (~210 mb) + cross-encoder (~80 mb) download once.
- **first index build per pdf**: rendering pages at 3000 px, table/figure detection, embedding every chunk on cpu.
  all cached after, so rebuilds with the same docs only re-embed.
- **streamlit itself**: every click reruns the script (now ~60 ms) and the browser re-renders.

## knobs

- `EVAL_WORKERS` (default 2): questions answered at once in evals. raise on paid tiers, lower if you hit 429s.
- `LLM_WORKERS` (default 3): parallel vision calls during index builds.
- candidate pool (settings): smaller pool = faster cross-encoder rerank.
- self-check `auto` instead of `full`: skips the verify call when confidence is already high.
