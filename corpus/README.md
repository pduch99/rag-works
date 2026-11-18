# corpus

one folder per knowledge base. the app and cli find them automatically.

```
corpus/<kb>/
  docs/        the files. pdf, png/jpg/webp, txt, md
  gold.json    gold-standard questions (the test bed tab edits this)
  kb.json      display name + a hint for the vision model (optional)
  cache/       vision descriptions + page contexts, written by index builds. commit it, it saves api calls
```

## gold.json (v2)

```json
{"id": "q-07", "type": "chart_value", "split": "dev",
 "question": "How many students enrolled in 2026?",
 "expected_answer": "5,100.",
 "key_facts": ["5,100"],
 "answerable": true,
 "evidence": [{"doc": "report.pdf", "page": 3, "box": [0.1, 0.5, 0.6, 0.8]},
              {"doc": "report.pdf", "page": 3, "quote": "2026 5,100"}]}
```

- evidence says where the answer is. use `quote` (exact text from the page) for text answers, `box` (fractions of the page, 0,0 = top-left) for visual ones, or just doc + page.
- it never names a chunk, so every chunker is scored the same way.
- unanswerable questions: `"answerable": false`, `"evidence": []`, expected answer says the docs don't say.
- types: text_fact, exact_term, table, chart_value, diagram_label, visual_location, visual_spatial, visual_route, cross_page, cross_doc, comparison, unanswerable.
- split: tune on `dev`, run `test` once at the end.
- old v1 files (expected_docs / expected_pages / expected_regions) still load.

## metrostate

campus_map.pdf + 4 metrostate.edu pages as text. 13 fp6 questions + 1 unanswerable, all dev. the cache has the fp4 whole-page and fp5 nw/ne/sw/se prose descriptions, so `page` and `region` with prose descriptions build with no vision calls.

## testbed

the professor's 10-pdf multimodal test bed. see testbed/README.md.
