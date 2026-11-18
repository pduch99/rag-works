# RAG Works — Full Project Handoff for Claude

> This handoff captures the full working context of the RAG Works capstone project, including project goals, professor feedback, architecture decisions, code evolution, FP2–FP5 implementation details, evaluation findings, current direction, and planned FP6+ roadmap.
>
> Student: **Patrick Duch**  
> Course: **ICS 499 Capstone**  
> Project: **RAG Works**  
> Team name: **RAGtag**  
> Domain used so far: **Metro State University**  
> Repo: **https://github.com/pduch99/rag-works**

---

# 1. High-level project vision

RAG Works started as a set of Python reference implementations for different RAG architectures, using Metro State University data as a common domain.

The professor's original expectation was roughly:

- implement around 15 RAG patterns
- build working Python reference implementations
- use Metro State documentation / website / schedules as a common domain
- learn how the architectures differ
- demo something working every iteration
- eventually expose the implementations through a front end

Over time, the professor has pushed the project toward something broader and more useful:

## Current end-goal

**RAG Works should become a universal web-based multimodal RAG chatbot and experimentation workbench.**

Instead of being only a Metro State chatbot, it should allow a user to load their own corpus and build a chatbot over that material.

Potential use cases:

- Metro State chatbot
- company policy chatbot
- product manual chatbot
- class / research notes chatbot
- personal knowledge bot
- “Patrick bot” built from Patrick’s own resume, bio, project docs, writing, photos, etc.

The system should not pretend to literally *be* a person. It should answer about the uploaded knowledge base.

A concise end-product statement:

> **RAG Works is a web-based multimodal RAG chatbot and experimentation platform that allows users to ingest PDF and image collections, select chunking and retrieval strategies, ask questions, inspect retrieved evidence, receive cited text and image-supported answers, and evaluate system performance against a gold-standard test bed.**

---

# 2. Professor’s strongest recent guidance

The professor has repeatedly emphasized:

> “You need to create, produce and demonstrate something each week.”

He said Patrick was getting “stuck in analysis,” so every FP should now produce a concrete runnable experiment, observed outputs, and a demo.

The professor specifically told Patrick for FP4:

> “Please take a PDF with images. RAG it. Ask questions in the context of images. And see how AI is responding. Please focus on this specific goal for FP4.”

More recently, the professor also focused on these specific topics:

- **chunking enhancement techniques**
- create a **test bed**
- about **10 PDF documents with images**
- create **gold-standard questions and answers**
- evaluate chatbot performance against the gold standard
- explain what is being done to improve chatbot performance / accuracy
- answers should ideally be **explainable**
- answers should include **citations**
- responses may include **supporting images**
- final system should show **how RAG is working internally**
- show document / chunk / source / page / image / retrieval info
- create something reusable, not Metro-State-hard-coded
- overall system should make RAG behavior visible and measurable

The professor also asked how “Library” becomes a chunk and wanted Patrick to explain the layers / flow visually.

---

# 3. Development philosophy going forward

The strongest framing now is:

> Every FP should build one new piece, test it against a controlled set, observe the result, and use the failures to motivate the next FP.

Ideal rhythm:

```text
Build
↓
Run
↓
Measure
↓
Find failure
↓
Improve
↓
Run same tests again
```

This is better than implementing disconnected RAG patterns with no shared test bed.

---

# 4. Current technical stack

## Retrieval / storage

- **ChromaDB**
- persistent local Chroma client
- local embedding model used by Chroma
- earlier notes referred to Chroma’s default `all-MiniLM-L6-v2` style embedding behavior
- vector retrieval is local

## LLM provider

Patrick wants to remain on **Groq**, not OpenAI API.

### Text model

Used for Naive RAG:

```text
openai/gpt-oss-20b
```

via Groq.

### Vision / multimodal model

Used for image and PDF image experiments:

```text
qwen/qwen3.8-27b
```

via Groq.

## Environment variable

```text
GROQ_API_KEY=...
```

Never commit the real `.env`.

## Main Python packages

Current root `requirements.txt` should contain at least:

```text
groq
chromadb
python-dotenv
pymupdf
pillow
```

Earlier project dependencies also included:

```text
requests
beautifulsoup4
```

for Metro State web page downloads.

---

# 5. Current repo structure

The project has evolved roughly toward:

```text
rag-works/
│
├── patterns/
│   ├── 01_naive_rag/
│   ├── 02_multimodal_rag/
│   ├── 03_pdf_image_rag/
│   └── 04_region_image_rag/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── rag-architecture-patterns.md
```

Patrick also has a GitHub-connected copy / repo at:

```text
https://github.com/pduch99/rag-works
```

At one point the ZIP copy had no `.git` folder, so Patrick cloned the repo separately and copied the edited files into the clone before pushing.

---

# 6. FP1 — scope clarification

FP1 was mostly scope clarification.

Key understanding from professor:

- primary goal: build Python reference implementations of different RAG architecture patterns
- roughly 1–2 patterns per week
- use Metro State information as the common domain
- learn how retrieval / complexity / use cases differ
- secondary goal: determine how to expose the implementations via UI

FP1 wording Patrick used:

> This week I focused on understanding and defining the scope of the RAG Works project. I reviewed the project requirements with the instructor and clarified that the primary goal is to develop Python reference implementations for different RAG architecture patterns.

Patrick also identified Metro State users as:

- students
- prospective students
- faculty
- staff

---

# 7. FP2 — Naive RAG

## Purpose

Build first working RAG reference implementation.

## Data

Metro State pages:

```text
registration.txt
admissions.txt
student_support.txt
academic_calendar.txt
```

Original `download_data.py` used:

- `requests`
- `BeautifulSoup`
- page cleaning
- save plain text

Known URLs used:

```text
https://www.metrostate.edu/academics/registration
https://www.metrostate.edu/apply/undergraduate-faqs
https://www.metrostate.edu/students/support
https://www.metrostate.edu/academics/calendar/fall-2026
```

## FP2 pipeline

```text
Metro State text files
↓
chunk text
↓
Chroma embeddings
↓
question
↓
semantic retrieval
↓
top 3 chunks
↓
Groq text model
↓
answer
```

## Chunking

Original implementation used approximately:

```text
size = 1000 characters
overlap = 200 characters
```

## FP2 model

Current intended Groq model:

```text
openai/gpt-oss-20b
```

## Important FP2 failure

Question:

> “When is registration deadline?”

Retrieved:

- registration.txt chunk 2
- registration.txt chunk 1
- academic_calendar.txt chunk 1

Generated answer incorrectly returned semester end dates such as:

- Spring 2027: May 3, 2027
- Summer 2027: August 13, 2027
- Fall 2027: December 12, 2027

This became a useful example of:

```text
retrieval can be related
but generation can still interpret evidence incorrectly
```

This distinction becomes important later in FP5.

---

# 8. FP3 — early multimodal exploration

Professor wanted Patrick to focus on **Multimodal RAG** rather than Advanced RAG.

Early FP3 thinking included:

- text chunks
- images
- PDFs
- visual descriptions
- original image paths
- retrieve visual source before generation
- pass original image to vision model

Important conceptual distinction established:

```text
image → LLM
```

is only vision.

Whereas:

```text
question
↓
retrieve image from knowledge base
↓
send retrieved image to vision model
↓
answer
```

is actual multimodal RAG.

FP3 was initially too analysis-heavy, which led directly to the professor’s FP4 feedback.

---

# 9. FP4 — PDF Image RAG

FP4 was the first very concrete visual RAG experiment and is important because it produced real before/after results.

## FP4 professor goal

Use a PDF with images, RAG it, ask questions in the context of the images, and observe AI behavior.

## Source PDF

Metro State campus map PDF.

Locally stored as:

```text
patterns/03_pdf_image_rag/data/source.pdf
```

The PDF had one page.

## FP4 folder

```text
03_pdf_image_rag/
├── data/
│   └── source.pdf
├── output/
│   ├── pdf_text.txt
│   ├── vision_description.txt
│   ├── results.json
│   └── pages/
│       └── page_1.png
├── prepare_pdf.py
├── vision_describe.py
├── pdf_rag.py
├── test_pdf_rag.py
└── README.md
```

---

# 10. FP4 — PDF preparation

## `prepare_pdf.py`

Used **PyMuPDF**.

The script:

1. opened the PDF
2. extracted normal PDF text
3. rendered page 1 as PNG
4. wrote text to:

```text
output/pdf_text.txt
```

5. wrote page image to:

```text
output/pages/page_1.png
```

The original script first used:

```python
import fitz
```

but this showed a deprecation warning.

It was later updated to:

```python
import pymupdf
```

The PDF extraction produced only about:

```text
1153 text characters
```

This included labels such as:

```text
LIBRARY AND LEARNING CENTER
STUDENT CENTER
PARKING RAMP
E. 7th Street
E. 6th Street
TUNNEL
SKYWAY
etc.
```

The key observation:

> PDF text extraction preserved the words, but mostly lost the visual / spatial relationships between them.

---

# 11. FP4 — vision description

## `vision_describe.py`

Used Groq:

```text
qwen/qwen3.8-27b
```

The rendered page image was base64 encoded and sent as:

```text
data:image/png;base64,...
```

Prompt focused on:

- building locations relative to each other
- streets
- parking areas
- tunnels
- skyways
- spatial relationships
- map legend information

Output stored at:

```text
output/vision_description.txt
```

The vision model was able to describe spatial information that plain PDF text extraction could not preserve.

---

# 12. FP4 — RAG design

## `pdf_rag.py`

FP4 indexed two coarse entries in Chroma:

```text
1. full extracted page text
2. full visual description of page 1
```

Metadata roughly:

```text
source = source.pdf
page = 1
type = text
```

or:

```text
source = source.pdf
page = 1
type = visual
```

This means FP4 chunking was **page-level**, not object-level.

For a query like:

> “Where is the Library?”

the system did not have a dedicated Library chunk.

Instead:

```text
question
↓
Chroma compares question against:
  - full page text
  - full page visual description
↓
retrieve most relevant entry / entries
↓
if visual entry retrieved:
  load original page_1.png
↓
Groq receives:
  question
  retrieved text
  retrieved visual description
  original image
↓
answer
```

Important explanation for professor:

> **Chroma does retrieval. Groq does visual interpretation and final answer generation.**

---

# 13. FP4 — controlled comparison

Patrick then modified `pdf_rag.py` to compare:

## Mode 1 — Text-only RAG

Only raw extracted PDF text.

## Mode 2 — Multimodal RAG

Retrieved text + visual representation + actual page image.

This enabled direct comparison using the same questions.

---

# 14. FP4 — fixed test set

`test_pdf_rag.py` used 8 fixed questions:

1. Where is the Student Center based on the map?
2. Where is the Library and Learning Center located?
3. Where is the parking ramp relative to the Student Center?
4. Where is St. John's Hall relative to the Student Center?
5. What building is south of E. 6th Street?
6. What parking areas are near the Library?
7. How would you get from the Student Center to the Library?
8. What tunnels or skyways are shown on the map?

Results were saved to:

```text
output/results.json
```

---

# 15. FP4 — key real findings

## Student Center question

Text-only response was vague and inferred proximity based on text order.

Multimodal response gave spatial relationships.

However, some multimodal responses had contradictions or questionable directionality.

## Library location

Text-only could identify the Library label but could not determine specific placement.

Multimodal could describe:

- north of E. 7th Street
- nearby Library parking
- relation to Maria Avenue
- surrounding visual context

## Parking ramp relative to Student Center

Text-only:

```text
I could not determine it from the text.
```

Multimodal:

provided a spatial relationship.

This was a clear multimodal win.

## Student Center → Library route

Text-only:

```text
I could not determine it from the text.
```

Multimodal:

attempted step-by-step directions based on the map.

## Tunnels / skyways

Multimodal answer became much more detailed, but introduced visual interpretations that needed manual verification.

This was a useful example that:

> multimodal answers can become more informative **and** more hallucination-prone.

---

# 16. FP4 main conclusion

Patrick’s strongest FP4 conclusion:

> **Text-only RAG worked well when the answer depended mainly on labels or words contained in the PDF. It struggled when the question required understanding spatial relationships. Adding the actual PDF image improved spatial reasoning, but multimodal generation also introduced new opportunities for incorrect or overconfident visual interpretation.**

This became the motivation for FP5.

---

# 17. FP4 evaluation categories

Professor asked about evals.

Patrick’s current eval framing:

1. **Retrieval accuracy**
   - did it retrieve correct document/page/type?

2. **Answer correctness**
   - correct / partial / wrong

3. **Grounding**
   - supported by retrieved evidence?

4. **Multimodal win rate**
   - did image-aware answer improve over text-only?

5. **Hallucination count**
   - unsupported details?

6. **Latency**
   - optional but useful

Eventually gold-standard eval data should contain:

```json
{
  "question": "...",
  "expected_answer": "...",
  "expected_source": "...",
  "expected_page": 1,
  "expected_region": ["ne", "se"],
  "question_type": "visual_spatial"
}
```

---

# 18. FP5 — region-level visual RAG

FP5 addresses a specific FP4 weakness:

> FP4 always treated the whole page as one visual source.

FP5 splits the page into smaller visual regions.

## FP5 goal

```text
PDF page
↓
split into regions
↓
describe each region
↓
store each region separately
↓
question
↓
retrieve most relevant region(s)
↓
send only relevant crop(s) to Groq
↓
answer
```

---

# 19. FP5 folder

```text
patterns/04_region_image_rag/
├── data/
│   └── source.pdf
├── output/
│   ├── pages/
│   ├── regions/
│   ├── region_descriptions.txt
│   └── results.json
├── prepare_regions.py
├── region_rag.py
├── test_region_rag.py
└── README.md
```

At the time of this handoff:

- `prepare_regions.py` exists and has run successfully
- `region_rag.py` exists and is running
- `test_region_rag.py` has not yet been fully finalized
- formal FP5 eval automation is the immediate next backend task

---

# 20. FP5 — `prepare_regions.py`

The current experiment splits each rendered page into four geometric quadrants:

```text
NW
NE
SW
SE
```

For each region:

1. crop image
2. save region image
3. send crop to Groq vision
4. get short description
5. store description in Chroma
6. attach metadata

Example metadata:

```text
source = source.pdf
page = 1
type = visual_region
region = ne
path = output/regions/page_1_ne.png
```

The full page text is also stored separately.

---

# 21. FP5 region outputs

Actual crops created:

```text
page_1_nw.png
page_1_ne.png
page_1_sw.png
page_1_se.png
```

The region descriptions were saved to:

```text
output/region_descriptions.txt
```

Observed region content:

## NW

Correctly contained / described:

- Library and Learning Center
- North Library Parking Lot
- Library Parking Lot
- 7th Street Parking Lot
- Grow-It Center
- Fine Arts Studio
- part of St. John’s Hall
- Labyrinth
- E. 7th Street

## NE

Correctly focused on:

- Student Center
- E. 7th Street
- Student Center amenities
- nearby paths / transit

## SW

Contained:

- New Main
- Founders Hall
- Auditorium
- Jason R. Carter Science Education Center
- E. 6th Street
- Mounds Blvd
- Courtyard parking
- tunnels / skyway

## SE

Contained:

- Parking Ramp
- Ramp Parking Lot
- Bates Avenue
- Maria Avenue
- E. 6th Street
- compass rose

---

# 22. FP5 — first retrieval success

Question:

> “Where is library located?”

Retrieved:

```text
source.pdf | page 1 | visual_region | region nw
source.pdf | page 1 | text | region -
source.pdf | page 1 | visual_region | region ne
```

This was a strong success:

> **NW ranked first, which is correct.**

The generated answer correctly identified the Library as being in the northwest / upper-left campus area.

This was the first concrete evidence that region-level retrieval can localize visual evidence better than whole-page retrieval.

---

# 23. FP5 — parking ramp retrieval experiment

Question:

> “Where is the parking ramp relative to the Student Center?”

Earlier with top-3 retrieval, Chroma returned:

```text
NE
text
NW
```

This missed SE, even though:

- Student Center is in NE
- Parking Ramp is in SE

This exposed a retrieval problem.

The retrieval function was modified to retrieve all 5 stored candidates:

```text
1 text chunk
4 visual regions
```

and print distances.

---

# 24. Groq image / token constraints discovered

## Image count limit

Groq vision request failed when too many region images were sent:

```text
Too many images provided. This model supports up to 3 images
```

This led to a design change:

- retrieve all candidates for evaluation
- send only a limited subset of top visual regions to Groq

## Input token limit

Another error occurred:

```text
Request too large
Limit 7000 input tokens
Requested 8560
```

This led to another improvement:

- retrieve all candidates for evaluation
- only send:
  - top 2 visual regions
  - top 1 text chunk
- keep generation context smaller

This distinction is useful for the final system:

> retrieval can inspect many candidates, while generation should receive only the most useful evidence.

---

# 25. FP5 — key parking ramp result

After retrieving all 5 candidates, actual ranking was:

```text
NE visual region   distance 0.6214
text               distance 0.7451
SE visual region   distance 0.8071
NW visual region   distance 0.8106
SW visual region   distance 0.8817
```

Selected for Groq:

```text
NE
SE
text
```

This means:

```text
Student Center region = retrieved correctly
Parking Ramp region   = retrieved correctly
```

So retrieval succeeded.

However, Groq generated the wrong spatial relationship and said the parking ramp was northwest of the Student Center.

The actual visual relationship is approximately south / southeast.

This became the most important FP5 finding:

```text
Retrieval = correct
Generation / visual reasoning = incorrect
```

This is a strong capstone insight:

> **retrieval accuracy and answer accuracy are separate problems.**

Even when the correct evidence is retrieved, the LLM can still reason incorrectly over the evidence.

---

# 26. FP5 evaluation direction

For each question, FP5 should now explicitly record:

```text
question

expected page
expected region(s)

retrieved page
retrieved region(s)

retrieval pass/fail

expected answer
generated answer

answer correct / partial / wrong

grounded yes/no

notes
```

Example:

```text
Question:
Where is the parking ramp relative to the Student Center?

Expected regions:
NE + SE

Retrieved:
NE + SE

Retrieval:
PASS

Expected answer:
Parking ramp is south/southeast of Student Center

Generated:
Parking ramp is northwest

Answer:
FAIL
```

This is exactly the kind of evaluation the professor wants.

---

# 27. Current FP5 architecture explanation

Use this explanation when presenting:

```text
PDF
↓
extract page text
↓
render page as image
↓
split page into NW / NE / SW / SE
↓
Groq describes each crop
↓
store text + region descriptions in Chroma
↓
user asks question
↓
Chroma ranks text + visual regions
↓
retrieve top candidates
↓
select best regions for generation
↓
send cropped images + text to Groq
↓
answer
```

Important responsibilities:

```text
Chroma = retrieval
Groq   = vision interpretation + answer generation
```

---

# 28. FP5 visual design

Patrick created / requested simple visual flow diagrams showing:

1. Input
2. Document Processing
3. Knowledge Creation
4. Retrieval + Answering
5. Evaluation

The ideal presentation diagram is simple and uses arrows rather than formal enterprise architecture boxes.

FP5 comparison:

```text
FP4:
whole-page visual retrieval

FP5:
region-level visual retrieval
```

Goal:

```text
improve retrieval precision
reduce visual confusion
```

---

# 29. Current end-product UI vision

Patrick wants to move the UI up sooner rather than leaving it until the very end.

The target is a visual chatbot / RAG workbench.

A mockup was generated with:

- left panel: uploaded documents
- center: chat
- right: retrieval trace
- bottom: eval metrics / comparison
- top controls:
  - knowledge base
  - chunking strategy
  - retrieval mode
  - run eval

Possible tabs:

```text
Chat
Documents
Chunks
Retrieval
Evals
Compare
```

The chatbot response should ideally include:

```text
answer
explanation
citations
source document
page
region
retrieved image(s)
retrieved text chunk(s)
similarity / distance score
```

For visual questions, response should be able to show the actual supporting crop(s).

Example:

```text
Answer:
The Student Center is north of the parking ramp.

Citations:
[Campus Map, p.1, NE]
[Campus Map, p.1, SE]

Supporting evidence:
[NE crop]
[SE crop]
```

---

# 30. Universal RAG scope

The system should not be Metro-State-only.

Metro State becomes the first test corpus.

A broader system should support:

```text
PDFs
text inside PDFs
images inside / rendered from PDFs
standalone images (optional)
```

Do not over-scope yet into:

- Word
- Excel
- PowerPoint
- video
- audio
- SQL
- agents
- OCR-heavy scans

Those can be future work.

The universal flow:

```text
user uploads corpus
↓
RAG Works ingests documents
↓
user selects strategy
↓
system builds knowledge base
↓
user asks questions
↓
RAG retrieves evidence
↓
LLM answers
↓
UI shows answer + evidence + citations
↓
eval compares result against gold standard
```

---

# 31. New problem statement

A strong current problem statement:

> **Current RAG systems can answer questions over documents, but it is often difficult to know whether the system retrieved the correct evidence, whether visual information was lost during preprocessing, whether chunking strategy affected retrieval quality, and whether generated answers are actually supported by the source material.**
>
> **RAG Works is a configurable multimodal RAG test bed that ingests PDF collections containing text and images, retrieves relevant evidence, generates cited answers, displays supporting source images, and evaluates performance against a gold-standard question-and-answer set.**

---

# 32. Test bed direction

Professor explicitly asked for:

> **10 PDF documents with images**

The system should create a controlled test bed around those.

Each test bed should have:

- document collection
- known answerable questions
- expected answers
- expected source document
- expected page
- expected image / region when relevant
- question category

Possible categories:

```text
text fact
visual spatial
table / chart
cross-page
multi-document
exact keyword
semantic question
route / relationship
```

A gold-standard entry may look like:

```json
{
  "question": "Where is the Student Center relative to the parking ramp?",
  "expected_answer": "The Student Center is north of the parking ramp.",
  "expected_source": "campus_map.pdf",
  "expected_page": 1,
  "expected_region": ["ne", "se"],
  "type": "visual_spatial"
}
```

---

# 33. Gold-standard evaluation metrics

The system should eventually calculate:

```text
document retrieval accuracy
page retrieval accuracy
region retrieval accuracy
answer accuracy
grounding accuracy
citation accuracy
image support accuracy
hallucination count / rate
latency
```

Potential comparison table:

```text
Strategy             Retrieval   Answer   Grounding   Citation
Fixed-size             ...
Page-level             ...
Region-level           ...
Semantic               ...
Hybrid + rerank        ...
```

Only actual measured numbers should be shown.

---

# 34. Chunking research direction

The professor specifically mentioned:

> **chunking enhancement techniques**

Current progression already provides multiple chunking strategies:

## FP2

```text
fixed-size text chunks
1000 characters
200 overlap
```

## FP4

```text
whole-page text
whole-page visual chunk
```

## FP5

```text
page text
NW / NE / SW / SE visual regions
```

Next chunking strategies to research / implement:

### Fixed-size chunking

Baseline.

### Page-level chunking

One PDF page = one chunk.

### Region-level visual chunking

Current FP5.

### Semantic chunking

Split by changes in meaning / topic rather than fixed size.

### Structure-aware chunking

Split based on:

- headings
- sections
- tables
- figures
- captions
- lists

### Parent-child chunking

Retrieve a small precise chunk but pass larger parent context to model.

### Contextual chunking / contextual retrieval

Add document / section context to chunks before embedding.

### Vision / layout-aware chunking

Instead of blindly cutting quadrants, detect meaningful regions:

- table
- chart
- map
- figure
- heading block
- paragraph group

A strong research question:

> **How does chunking strategy affect retrieval accuracy and answer quality across different question types?**

---

# 35. Other accuracy / performance techniques to explore

Potential future experiments beyond what professor directly showed:

## Hybrid retrieval

Combine:

```text
vector semantic retrieval
+
BM25 keyword retrieval
```

Useful for exact labels / names / IDs and semantic queries.

## Reranking

Retrieve many candidates, then rerank before generation.

Example:

```text
retrieve top 10
↓
reranker
↓
send best 3 to LLM
```

This directly addresses wrong regions ranking too high.

## Query rewriting

Rewrite vague user questions into better retrieval queries.

## Query decomposition

Example:

```text
Where is parking ramp relative to Student Center?
↓
find Student Center
find Parking Ramp
compare locations
```

## Corrective RAG

Evaluate retrieved evidence and retry if evidence is weak.

## Answer verification

After generation:

```text
Does this answer actually follow from the retrieved evidence?
```

This is especially relevant to the FP5 case where retrieval succeeded but generation failed.

## Metadata-aware retrieval

Use filters such as:

```text
document
page
region
content type
date
```

## Direct visual embeddings

Current visual retrieval is:

```text
image
↓
Groq-generated text description
↓
embed description
↓
retrieve
```

Potential future improvement:

```text
image
↓
visual embedding
↓
retrieve image directly
```

Candidate technologies:

- CLIP
- SigLIP
- ColPali-style page retrieval

## Multi-page multimodal retrieval

Retrieve correct page(s) across a larger multi-page PDF.

---

# 36. Revised roadmap from FP6 onward

Patrick wants the UI moved up earlier.

The current proposed roadmap:

## FP6 — Web Chatbot UI + Retrieval Trace

Build first usable RAG Works web interface.

Recommended technology:

```text
Streamlit
```

because:

- Python
- easy to connect existing code
- fast to demo
- avoids separate frontend/backend complexity initially

FP6 UI should show:

- chat input
- generated answer
- citations
- retrieved text chunks
- retrieved image crops
- document/page/region metadata
- similarity/distance
- which chunks/images were actually sent to Groq
- document upload
- simple eval screen / button

Basic flow:

```text
user asks
↓
RAG retrieves
↓
UI shows retrieved chunks/images
↓
Groq generates
↓
UI shows answer + citations + evidence
```

The UI becomes the shell for all later experiments.

---

## FP7 — Chunking Test Bed

Add selectable chunking strategies:

```text
Fixed-size
Page-level
Region-level
Semantic
```

UI dropdown:

```text
Chunking Strategy:
[ Region-level ▼ ]
```

Run same gold-standard questions against each.

Measure:

```text
document retrieval accuracy
page retrieval accuracy
region retrieval accuracy
answer correctness
grounding
citation correctness
latency
```

---

## FP8 — Hybrid Retrieval + Reranking

Compare:

```text
Vector only
Hybrid
Hybrid + reranking
```

Architecture:

```text
question
↓
vector retrieval
+
BM25 retrieval
↓
combine
↓
rerank
↓
best evidence
```

---

## FP9 — Better Multimodal Retrieval

Compare:

```text
caption-based visual retrieval
vs
direct visual embedding retrieval
```

Candidate model / method:

- ColPali
- CLIP
- SigLIP

Goal:

> determine whether direct visual retrieval finds correct page/region more reliably than caption-based retrieval

---

## FP10 — Gold Standard Evaluation Dashboard

Build larger test bed:

```text
~10 PDFs with images
30–50 gold-standard questions
```

Show metrics in UI.

---

## FP11 — Corrective / Self-checking RAG

Add:

```text
retrieve
↓
grade evidence
↓
generate
↓
verify answer
↓
retry / correct if needed
```

This directly targets:

```text
correct retrieval
but wrong answer
```

---

## FP12+ — final integration / polish

Final tabs:

```text
Chat
Documents
Chunks
Retrieval
Evals
Compare
```

---

# 37. Ideal final UI behavior

A user uploads:

```text
10 PDFs
4 images
3 text documents
```

The system processes them.

User asks:

> “Where is the Student Center relative to the parking ramp?”

RAG Works should show:

## Answer

```text
The Student Center is north of the parking ramp.
```

## Explanation

```text
The retrieved map regions show the Student Center in the NE crop and the parking ramp in the SE crop.
```

## Citations

```text
[Campus Map, page 1, NE]
[Campus Map, page 1, SE]
```

## Supporting images

Display the actual NE + SE crops.

## Retrieval trace

```text
1. page 1 NE   distance 0.6214
2. page 1 text distance 0.7451
3. page 1 SE   distance 0.8071
...
```

## Diagnosis

If wrong:

```text
Retrieval:
PASS

Generation:
FAIL

Reason:
Correct visual evidence was retrieved, but the model inferred the spatial direction incorrectly.
```

---

# 38. What “explainable” means in this project

When professor asks:

> “Is it explaining its answer?”

The goal is not hidden chain-of-thought.

The system should provide:

- short evidence-based explanation
- source citations
- retrieved chunk previews
- supporting images
- page / region metadata
- retrieval scores
- pipeline trace

This is enough to make the result auditable without exposing private model reasoning.

---

# 39. What “citations” should mean

For text:

```text
[student_handbook.pdf, p.12]
```

For visual evidence:

```text
[campus_map.pdf, p.1, NE]
```

Clickable / expandable UI should ideally show:

- source document
- page number
- region
- extracted text
- original crop/image

---

# 40. What “responses contain images” should mean

When visual evidence supports an answer, the response should show:

```text
Answer text

Source citation

Supporting image crop
```

Not every answer needs an image.

Only visual questions / visual evidence should return images.

---

# 41. Current immediate implementation priority

Patrick has already built the core FP5 region retrieval prototype.

Immediate next technical items:

1. finish `test_region_rag.py`
2. formalize expected regions / answers
3. save FP5 eval results
4. add manual grades:
   - retrieval pass/fail
   - answer correct/partial/wrong
   - grounded yes/no
5. build FP6 Streamlit UI using current region-level RAG
6. expose:
   - chat
   - source citations
   - retrieval trace
   - retrieved images
   - distances
7. begin building universal document ingestion rather than Metro-State-hard-coded paths

---

# 42. Recommended FP6 implementation scope

Do **not** overbuild FP6.

Version 1 should simply wrap the existing backend.

Suggested files:

```text
rag-works/
├── app.py
├── rag/
│   ├── retrieval.py
│   ├── generation.py
│   └── citations.py
├── patterns/
│   └── existing FP folders
├── data/
├── evaluations/
└── requirements.txt
```

Initially `app.py` can even import current region RAG code directly.

Later refactor common logic into `/rag`.

Streamlit UI sections:

```text
Sidebar:
- knowledge base
- upload files
- chunking mode
- top-k

Main:
- chat
- answer

Right / expandable panel:
- retrieved chunks
- source page
- region
- distance
- images

Bottom / tabs:
- Chat
- Documents
- Chunks
- Retrieval
- Evals
- Compare
```

---

# 43. Important lesson from current work

Do not conflate:

```text
retrieval quality
```

with:

```text
answer quality
```

Current FP5 proved:

```text
correct regions can be retrieved
while
final answer can still be wrong
```

This distinction should become central to RAG Works.

Evaluation needs to independently score:

```text
retrieval
generation
grounding
citation
```

---

# 44. Presentation style professor seems to prefer

Avoid long theoretical explanations.

Lead with:

> “Here is what I built.”

Then show:

1. source PDF / test corpus
2. processing code
3. chunks
4. question
5. retrieval result
6. answer
7. expected result
8. pass/fail
9. what changed from last FP
10. what next FP will improve

Every weekly demo should have:

```text
artifact
code
test
result
finding
next experiment
```

---

# 45. Useful one-liners for meetings

## Explain current FP5

> “FP4 treated the whole PDF page as one visual chunk. FP5 splits each page into smaller visual regions, describes and indexes each region separately, and retrieves only the most relevant regions for the question.”

## Explain Chroma vs Groq

> “Chroma is responsible for retrieval. Groq is responsible for interpreting the retrieved evidence and generating the answer.”

## Explain current chunking

> “The page text is one text chunk, and each NW/NE/SW/SE crop is a separate visual chunk with page and region metadata.”

## Explain Library example

> “For the Library question, Chroma ranked the NW region first, which is correct because that is where the Library appears.”

## Explain parking-ramp failure

> “The system successfully retrieved both the Student Center and Parking Ramp regions, but the vision model still gave the wrong direction. That showed me retrieval accuracy and generation accuracy need to be evaluated separately.”

## Explain universal end goal

> “RAG Works is becoming less of a single Metro State chatbot and more of a RAG experimentation workbench where a user can upload a corpus, ask questions, inspect retrieved evidence, get cited text and image-supported answers, and compare different chunking and retrieval strategies against a gold standard.”

---

# 46. Things not to overclaim

Do not claim:

- multimodal always improves answers
- region chunking is already better overall
- FP5 answer accuracy improved globally
- retrieval is solved
- visual descriptions are ground truth
- current metrics are statistically meaningful
- current system is universal yet
- current app already supports 10 PDFs
- direct visual embeddings have been implemented
- semantic chunking is already implemented
- hybrid retrieval is already implemented
- reranking is already implemented
- citations UI exists already
- Streamlit UI exists already

Those are planned / future work unless implemented later.

---

# 47. Coding style preferences

Patrick prefers:

- simple Python
- compact code
- sparse comments
- minimal blank lines
- no giant tutorial docstrings
- full file replacements when making changes
- step-by-step execution when coding
- one command at a time when setup is fragile
- quick pace when simple

---

# 48. Git / repo notes

Earlier local project path:

```text
C:\Users\pduch\Downloads\rag-works-fp3\rag-works
```

GitHub clone path used during recovery:

```text
C:\Users\pduch\Downloads\rag-works-fp3\rag-works\rag-works-github
```

The non-git ZIP working copy originally gave:

```text
fatal: not a git repository
```

Fix used:

```text
git clone https://github.com/pduch99/rag-works.git rag-works-github
```

then `robocopy` from edited folder into the Git clone, excluding:

```text
.git
.venv
__pycache__
chroma_db
.env
```

then:

```text
git add .
git commit -m "Add multimodal and PDF image RAG work"
git push origin main
```

Never commit `.env`.

---

# 49. Recommended `.gitignore`

At minimum:

```text
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

Later consider ignoring large generated artifacts / caches if needed.

---

# 50. Proposed universal architecture

A clean high-level architecture:

```text
                    RAG WORKS
┌─────────────────────────────────────────────┐
│               USER / TEST BED               │
│ Upload docs | Ask question | Run eval       │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│             DOCUMENT INGESTION              │
│ PDFs | Text | Images                        │
│ extraction | rendering | metadata           │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│                 CHUNKING                    │
│ fixed | page | region | semantic | layout   │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│               KNOWLEDGE BASE                │
│ Chroma | embeddings | text | images         │
│ metadata | source paths                     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│                 RETRIEVAL                   │
│ vector | hybrid | rerank | filters          │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│                GENERATION                   │
│ question + text + images → Groq             │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│               EXPLANATION                   │
│ answer | citations | chunks | images        │
│ page | region | score | trace               │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│                EVALUATION                   │
│ retrieval | answer | grounding | citation   │
│ image accuracy | latency | hallucination    │
└─────────────────────────────────────────────┘
```

---

# 51. North-star product concept

The strongest simple summary:

> **RAG Works is a chatbot on the surface and a RAG laboratory underneath.**

The user experiences:

```text
upload documents
ask questions
get answers
```

The developer / evaluator can inspect:

```text
what was chunked
what was embedded
what was retrieved
what image was selected
what page/region it came from
what went into the model
whether the answer was correct
whether the answer was grounded
which strategy performed best
```

That is the central identity of the project going forward.

---

# 52. Immediate handoff to next assistant / Claude

When continuing from this handoff:

1. Treat **FP5 region-level RAG** as the current implemented backend.
2. Do not assume `test_region_rag.py` is fully done.
3. Do not assume the Streamlit UI exists yet.
4. The next major build should likely be **FP6 web UI**, unless Patrick first wants to finish FP5 eval automation.
5. Preserve Groq as the provider.
6. Preserve Chroma as current vector store.
7. Keep Metro State as the main test corpus while making code more universal.
8. Build toward:
   - source citations
   - supporting images
   - retrieval trace
   - chunk inspection
   - gold-standard evals
9. Keep each FP demonstrable and concrete.
10. Prefer experiments with measurable before/after results.

---

# 53. Suggested next coding tasks

If Patrick asks “what next?”:

## Option A — finish FP5 cleanly

Build:

```text
patterns/04_region_image_rag/test_region_rag.py
```

Include:

```text
question
expected page
expected regions
retrieved regions
retrieval pass/fail
expected answer
generated answer
manual grade
grounded
```

Save:

```text
output/results.json
```

## Option B — begin FP6 UI

Install:

```text
streamlit
```

Create:

```text
app.py
```

Version 1:

- chat input
- call current `region_rag.ask()`
- display answer
- display retrieval list
- display region images
- display page / region / distance
- citations in plain text

Then improve styling later.

---

# 54. Long-term research questions RAG Works can answer

Potential final-report research questions:

1. How does chunking strategy affect retrieval accuracy?
2. How does visual chunking affect spatial question performance?
3. Does region-level retrieval outperform whole-page retrieval?
4. Does hybrid retrieval improve document / page / region selection?
5. Does reranking improve retrieval precision?
6. Does direct visual embedding retrieval outperform caption-based retrieval?
7. Does answer verification reduce unsupported responses?
8. Which question types benefit most from multimodal RAG?
9. How often does correct retrieval still lead to incorrect generation?
10. How effective are citations and supporting images at making RAG output auditable?

These questions fit the professor’s direction much better than “which chatbot is best?”

---

# 55. Final key insight

The project is no longer best described as:

```text
build a Metro State chatbot
```

It is better described as:

```text
build a universal, inspectable, measurable multimodal RAG chatbot platform
```

Metro State is the first controlled test bed.

The project should demonstrate:

```text
ingestion
chunking
retrieval
multimodal evidence
generation
citations
explanation
evaluation
comparison
improvement
```

The UI should make these visible.

That is the current north star for RAG Works.
