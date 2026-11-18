# FP10 plan: ship it + final results

## make it easy for anyone to try
- [ ] bring-your-own-key box in the app (session only, never saved) so a hosted copy works for anyone with a free gemini key
- [ ] deploy to streamlit community cloud from the github repo
- [ ] demo mode with no key: the sample kb ships with saved answers so visitors can click through answers, traces and evals
- [ ] setup.bat / setup.sh: venv, install, copy .env
- [ ] pin exact versions in requirements.txt
- [ ] commit corpus/*/cache so a first build of the sample kb makes no vision calls
- [ ] readme: screenshots, a short gif, "try it in 2 minutes", results chart
- [ ] license (mit) + tagged release

## learn tab
- [ ] anatomy of rag, shown on a live trace
- [ ] the fp2 -> fp10 story: each failure and the fix it led to
- [ ] one card per technique: what, why, cost, measured result, "try it" button

## final results (test split, once)
- [ ] ladder: fp6 baseline -> best fp7 index -> + hybrid -> + rerank -> + labels -> + self-check
- [ ] leave-one-out: full minus each piece
- [ ] by question type, plus model calls and time per question
- [ ] judge agreement on ~20 hand-checked answers

## presentation
- [ ] story slides, live demo of one question through the whole trace, the numbers
