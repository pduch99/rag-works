# FP9: self-checking RAG + label positions

fp5 showed the right evidence can come back and the answer can still be wrong (the parking ramp). fp9 goes after the answer side.

## what changed

- **label positions**: the pdf text layer has every label's exact position. visual evidence now carries them,
  e.g. `STUDENT CENTER (SC) (68,44); PARKING RAMP STUDENT, EMPLOYEE AND GUEST PARKING (80,56)`.
  "where is the ramp relative to the student center" becomes comparing two pairs of numbers (x grows east, y grows south).
- **self-check** dropdown:
  - `none`: answer straight from the evidence (fp2-fp8)
  - `grade`: before answering, the text model marks which evidence is relevant and whether the question is answerable.
    irrelevant evidence is dropped. if nothing helps it retries once with a better search query, then says
    "I couldn't find this" instead of guessing
  - `verify`: after answering, the answer is split into claims and each is checked against its evidence, with the actual
    crops attached for visual claims. unsupported claims trigger one fix-up pass
  - `full`: grade + verify
  - `auto`: grade always, verify only when confidence is under 0.75 (cheaper full)
- **confidence** on every answer: retrieval score + share of evidence kept + share of claims that checked out
- ui: confidence chip, claims list with ✓/✗, the answer before the fix, "no answer" styling
- evals: abstained / hallucinated / false "not found" / corrected / model calls per question / our-code time

## theory in one line each

- corrective rag (crag, yan et al 2024): grade retrieved evidence, act on the grade (refine, re-search, abstain) before generating.
- claim verification (the idea behind ragas faithfulness): break the answer into atomic claims, check each against the context.
- selective answering: abstaining on low confidence trades a little coverage for a lot of trust. measure both sides:
  hallucinations on unanswerable questions and false "not found" on answerable ones.
- grounding positions in the source: the model guesses directions from pixels badly; the pdf already knows where every label is.

## experiment

`python cli.py ladder testbed fp9 --index <best fp7 index> --retriever hybrid --reranker cross --split dev`

1. no labels, no check
2. + label positions
3. + grade
4. + verify
5. full
6. auto

watch: generation failures, hallucinated (unanswerable answered anyway), false abstain, model calls per question.
the trade-off chart: answer score vs calls per question.
