# The RAG framework

RAG as seven stages. Every RAG type changes one or more of them. Each type is described with the same fields, so they can be compared. Implemented types map to switches in RAG Works and are measured on the same gold questions (`python story.py`). This file is generated from `rag/framework.py`.

## Stages

| # | stage | what it decides | switches in RAG Works |
|---|---|---|---|
| 1 | chunk | how documents become pieces: fixed text, pages, image regions, headings and tables, real figure boxes | `chunker` |
| 2 | describe images | turn each image crop into searchable text: free prose, or every label plus where it sits | `style` |
| 3 | embed | text to vectors. which model, and whether each chunk gets a line of page context first | `embedder`, `contextual` |
| 4 | retrieve | find candidates: by meaning (vector), by exact words (bm25), or both fused | `retriever` |
| 5 | rerank | reorder the candidates with a model that reads question and chunk together | `reranker` |
| 6 | build evidence | pick what the model sees and add facts code can work out: label positions, directions | `labels`, `spatial` |
| 7 | check + answer | answer with citations, optionally grade the evidence first and verify the claims after | `check` |

## Indexing

### Naive RAG

- **status:** implemented (fp2)
- **stages:** ingest, embed, retrieve, generate
- **problem it solves:** a model can't answer about documents it never saw
- **how it works:** cut text into fixed chunks, embed them, take the top matches by similarity, paste them into the prompt
- **cost:** 1 model call per question. cheapest to build and run
- **how it fails:** chunks split facts in half, images are invisible, near-duplicate passages confuse it
- **use it when:** plain text documents and simple fact lookups. the baseline everything else is measured against

### Multimodal RAG

- **status:** implemented (fp4-fp5)
- **stages:** ingest, describe
- **problem it solves:** answers live in maps, charts and diagrams, not in the text layer
- **how it works:** crop pages or regions into images, have a vision model describe each crop, index the descriptions, send the crops with the answer
- **cost:** 1 vision call per crop at build time (cached), images make every answer call bigger
- **how it fails:** descriptions miss small labels, a crop seam can cut a place in half, the model misreads directions
- **use it when:** any corpus with figures that matter: maps, slides, scanned forms, charts

### Structure-aware chunking

- **status:** implemented (fp7)
- **stages:** ingest, describe
- **problem it solves:** fixed-size chunks ignore the document: a table or a section gets cut mid-way
- **how it works:** chunk by headings and sections, keep tables whole as markdown, crop real figure boxes found in the pdf, date lists become one record per line
- **cost:** a one-time layout pass per page (cached). no extra model calls
- **how it fails:** heuristics break on odd layouts (a map's street grid looked like a table, calendar dates looked like headings)
- **use it when:** documents with real structure: handbooks, policies, reports, calendars
- **measured:** `fp6 baseline` → `fp7 better index` in story.py

### Contextual retrieval

- **status:** implemented (fp7)
- **stages:** embed
- **problem it solves:** a small chunk loses what it's about ('it opens at 8' — what does?)
- **how it works:** a model writes one line of page context, which goes in front of every chunk from that page before embedding
- **cost:** 1 text call per page at build time (cached)
- **how it fails:** a vague context line adds noise. measured only together with the other fp7 changes here
- **use it when:** short chunks from long documents, lots of pronouns or repeated section names
- **measured:** `fp6 baseline` → `fp7 better index` in story.py


## Retrieval

### Sparse / keyword retrieval

- **status:** implemented (fp8)
- **stages:** retrieve
- **problem it solves:** embeddings blur exact terms: codes, names, '60-89 CR'
- **how it works:** bm25 scores chunks by how often the question's exact words appear, weighted by how rare they are
- **cost:** about 1 ms, no model
- **how it fails:** misses paraphrases and synonyms, knows nothing about images
- **use it when:** ids, course codes, names, legal or technical wording

### Hybrid RAG

- **status:** implemented (fp8)
- **stages:** retrieve
- **problem it solves:** vector search misses exact terms, keyword search misses meaning
- **how it works:** run both and merge the rankings with reciprocal rank fusion (k=60): a chunk ranked well by either rises
- **cost:** about 5 ms more than vector alone, no model calls
- **how it fails:** if both rankers are wrong it can't help. fusion can push a good vector-only hit down
- **use it when:** almost always worth it. mixed questions, some exact and some fuzzy
- **measured:** `fp7 better index` → `fp8 hybrid + rerank` in story.py


## Post-retrieval

### Reranked RAG

- **status:** implemented (fp8)
- **stages:** rerank
- **problem it solves:** first-stage search is fast but rough. the best chunk is often 4th or 9th
- **how it works:** take a wider pool, then a cross-encoder reads question + chunk together and reorders (or a model ranks them, crops included)
- **cost:** cross-encoder about 250 ms on cpu for 10 candidates. the model version costs a call
- **how it fails:** text-only rerankers can't judge image crops, and it can only reorder what search found
- **use it when:** when recall is fine but the right chunk isn't on top
- **measured:** `fp7 better index` → `fp8 hybrid + rerank` in story.py

### Evidence-augmented RAG

- **status:** implemented (fp9-fp10)
- **stages:** evidence
- **problem it solves:** the right crop is retrieved but the model still flips north and south
- **how it works:** add facts code can compute: exact label positions from the pdf text layer (legend skipped), then directions between the places the question names
- **cost:** about 1 ms, no model calls
- **how it fails:** only works where a text layer exists. a small model can still contradict a fact it was handed
- **use it when:** maps, floor plans, diagrams, tables: anything where code can do the math better than the model
- **measured:** `fp9 labels + self-check` → `fp10 spatial facts` in story.py


## Generation control

### Corrective RAG

- **status:** implemented (fp9)
- **stages:** generate
- **problem it solves:** the model answers even when the evidence is off-topic, and guesses when nothing was found
- **how it works:** a grader marks which evidence is relevant. nothing relevant: search again with a better query, still nothing: say not found
- **cost:** 1 extra call per question, 3 when it retries
- **how it fails:** a weak grader drops the right evidence or retries into worse results (seen with the 8b local model)
- **use it when:** high stakes answers where 'I don't know' beats a wrong answer
- **measured:** `fp8 hybrid + rerank` → `fp9 labels + self-check` in story.py

### Self-verifying RAG

- **status:** implemented (fp9)
- **stages:** generate
- **problem it solves:** a fluent answer can include claims the evidence doesn't support
- **how it works:** after answering, split the answer into claims, check each against the evidence and the crops, rewrite once if any fail. auto = only when confidence is low
- **cost:** 1-2 extra calls, roughly doubles answer time
- **how it fails:** a checker built from the same small model shares its blind spots and approves its mistakes
- **use it when:** with a strong checker model, or when you need a claim-by-claim audit trail
- **measured:** `fp8 hybrid + rerank` → `fp9 labels + self-check` in story.py


## Architecture

### Visual document retrieval (ColPali)

- **status:** described (future work (byaldi / colpali-engine))
- **stages:** ingest, describe, embed, retrieve
- **problem it solves:** describing images in text loses detail before search even starts
- **how it works:** embed page images directly with a vision-language retriever (many vectors per page), search pages by image, no text extraction
- **cost:** needs a gpu to index, bigger index (many vectors per page)
- **how it fails:** returns whole pages, not exact regions. weaker on exact codes and numbers
- **use it when:** scanned or image-heavy corpora with no usable text layer

### GraphRAG

- **status:** described (future work)
- **stages:** ingest, retrieve
- **problem it solves:** questions about relationships across many documents ('which offices connect to...')
- **how it works:** a model extracts entities and relations into a graph, questions are answered by walking the graph and community summaries
- **cost:** many model calls to build the graph, slow to update
- **how it fails:** extraction errors become graph errors, overkill for single-fact lookups
- **use it when:** large connected corpora: org charts, case files, research literature

### Agentic RAG

- **status:** described (future work. corrective rag's retry is a one-step version)
- **stages:** retrieve, evidence, generate
- **problem it solves:** one search isn't enough for multi-step questions
- **how it works:** a model plans, picks tools (search, calculator, the spatial helper), searches several times and decides when it has enough
- **cost:** many calls per question, latency is unpredictable
- **how it fails:** loops, wanders off, hard to evaluate because every run takes a different path
- **use it when:** multi-hop questions, when tools exist and latency doesn't matter much
