# legacy (fp2 - fp5)

the original one-folder-per-pattern experiments, kept as-is for the report. the app in the repo root replaces all of them.

| folder | fp | what it showed |
|---|---|---|
| fp2_naive_rag | fp2 | text chunks 1000/200 + chroma + groq. "registration deadline" pulled related chunks but answered with semester end dates |
| fp3_multimodal_rag | fp3 | first image + text experiment |
| fp4_pdf_image_rag | fp4 | campus map: text-only vs whole page image. results in output/results.json |
| fp5_region_image_rag | fp5 | page cut into nw/ne/sw/se. library -> nw ranked first. parking ramp: ne + se retrieved, model still said north-west |

they still run. from the repo root, with .env filled in:

```
python legacy/fp5_region_image_rag/region_rag.py
```

fp4 and fp5 need the map copied back first: `corpus/metrostate/docs/campus_map.pdf` -> `legacy/fp5_region_image_rag/data/source.pdf` (same for fp4). each makes its own chroma_db/ inside its folder (gitignored). fp5's descriptions and fp4's whole-page description were copied into `corpus/metrostate/cache/` so the new app reuses them.
