# testbed

the professor's ask: ~10 pdfs with images, gold questions + answers, measure the chatbot against them.

## pdfs to collect (public domain or cc-licensed only)

| # | kind | tests | example source |
|---|---|---|---|
| 1 | campus map (copy campus_map.pdf from metrostate) | spatial | metro state |
| 2 | second map or floor plan | look-alike distractor, spatial | a metro state building / any university floor plan |
| 3 | transit or trail map | dense labels, routes | metro transit / a state park map |
| 4 | handbook or catalog excerpt | headings, policies, tables | metro state catalog |
| 5 | research paper with charts | chart values, captions | arxiv (cc-by) |
| 6 | government report with tables + bar charts | tables, numbers | a minnesota agency annual report |
| 7 | product manual with labeled diagrams | diagram labels, steps | any appliance manual pdf |
| 8 | slide deck saved as pdf | low text, layout | a public conference / class deck |
| 9 | infographic | visual-heavy | cdc / nasa infographics |
| 10 | form instructions with a flowchart | process, branching | irs / fafsa / dmv guides |

keep each 5-30 pages. at least 3 pages per pdf with real visuals. one look-alike pair helps test doc picking.

## gold questions (~45)

- ~3-4 per pdf, plus 5 cross-doc / comparison, plus 5 unanswerable
- mix: text_fact, exact_term (codes, names, numbers), table, chart_value, diagram_label, visual_location, visual_spatial
- every one checked against the page by you. the test bed tab's "draft with ai" is a starting point, not the answer
- evidence: quote for text answers, area for visual ones
- split ~1/3 dev, 2/3 test. only run test at the very end

## build + run

```
python cli.py ladder testbed fp7 --split dev
python cli.py ladder testbed fp8 --index layout.bge.structured.ctx --split dev
```

budget: every visual crop is one vision call the first time (cached after). ~10 pdfs x ~15 pages ≈ a few hundred calls for layout. spread builds over a couple of days if the free tier runs out.
