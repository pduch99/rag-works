# FP6: web workbench + built-in evals

## what changed from fp5

| fp5 | fp6 |
|---|---|
| one script per pattern, hard-coded `source.pdf` | one `rag/` core, any kb folder, any number of docs |
| quadrants only | text / page / region 2x2 / region 3x3, switchable |
| terminal printout | streamlit ui: chat, documents, chunks, evals, compare |
| answer text only | `[n]` citations, a "why" line, supporting crops |
| eval was manual and not finished | 13 gold questions, auto retrieval scoring, llm judge + manual override, saved runs |
| groq only | groq or gemini |
| sent whatever top-k returned, hit the 3-image and 7k-token limits | evidence selection that respects each provider's image cap and token budget |

Generation also changed. The prompt now numbers the evidence, says what nw/ne/sw/se mean on the page, says up = north unless a compass shows otherwise, and asks the model to locate both places before comparing them. The optional **whole page** toggle sends the full page next to the crops for orientation. Both target fp5's parking-ramp failure. Whether they fix it is something to measure, not claim.

## demo script (about 5 min)

1. **What I built.** Open the app. Top bar: knowledge base, chunking strategy, model, run eval. Below: documents | chat | retrieval trace.
2. **Documents tab (bottom).** Show the campus map with the 2x2 grid drawn on it, then the extracted text. This answers "how does Library become a chunk?"
3. **Chunks tab.** Click the nw visual chunk to show the crop and the description that actually got embedded.
4. **Chat.** Ask *where is the library?* Show the answer, the citation, the nw crop, and the retrieval trace with distances and which rows were sent.
5. **Chat.** Ask *where is the parking ramp relative to the student center?* Show whether ne + se were retrieved and what direction the model gave.
6. **Evals.** Run `region`, then `region` with the whole-page toggle, then `page` and `text`. Show the diagnosis column: retrieval failures vs generation failures.
7. **Compare.** Chart the runs. Use only the measured numbers.
8. **Next.** The 10-pdf test bed and fp7 chunking strategies on the same gold set.

## one-liners

- "Retrieval and answering are scored separately, so every miss says whose fault it was."
- "Every answer cites numbered evidence, and the UI shows the exact crops and the exact prompt the model saw."
- "Same gold questions, different strategy or provider, one click. That's how each FP proves it improved something."
