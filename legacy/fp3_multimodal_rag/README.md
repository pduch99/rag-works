# Multimodal RAG

FP3 expands the FP2 Naive RAG example so the knowledge base can contain both text and visual information.

## What changed from FP2

FP2 indexes Metro State text and retrieves text chunks before generation. FP3 keeps that same basic retrieval flow, but also prepares images for retrieval. A vision-capable model first creates a searchable description of each image. Chroma indexes that description with metadata pointing back to the original image.

When an image result is retrieved, the original image is attached to the generation request. This means the final model can inspect the visual source instead of answering only from its text description.

```text
Text documents -> chunks ------------------\
                                             -> Chroma -> top results
Images -> vision description -> metadata ---/
                                                   |
                                                   v
                                  load retrieved original image
                                                   |
                                                   v
                                  question + text + image -> LLM
                                                   |
                                                   v
                                                 answer
```

## Data

The example reuses the Metro State text documents from FP2:

- registration.txt
- admissions.txt
- student_support.txt
- academic_calendar.txt

It also includes the RAG Works final-vision infographic as a visual source. This gives the implementation questions that cannot be answered from the Metro State text files alone and makes the visual retrieval path easy to demonstrate.

## Run

From `patterns/02_multimodal_rag`:

```powershell
python prepare_data.py
python multimodal_rag.py
```

`prepare_data.py` needs to be run before querying. It indexes the text chunks and creates a searchable description of the image.

Retrieval-only tests can be run with:

```powershell
python test_multimodal_rag.py
```

## Example questions

Text questions:

- When do Fall 2026 classes begin?
- How do I register for classes?
- What student support resources are available?

Visual questions:

- What steps are shown in the Naive RAG architecture?
- What does the final RAG Works interface show for Advanced RAG?
- What is the long-term goal shown in the RAG Works visual?

## What I learned

Multimodal RAG adds another step to ingestion and retrieval. An image cannot be searched in the same way as the text chunks in this simple implementation, so a text description is generated and embedded for retrieval. The original image still needs to be preserved so it can be supplied to the model after retrieval.

This also showed an important difference between using a vision model and building multimodal RAG. Sending an image directly to an LLM demonstrates vision. In this implementation, the user's question first retrieves the relevant source from the knowledge base, and the retrieved image is then supplied to the model.

This is a simple reference implementation, not a complete multimodal architecture. Image retrieval currently depends on generated text descriptions rather than a joint image/text embedding model. That is a limitation worth comparing with later patterns.

## Next

The next iteration can use a consistent evaluation set across FP2 and FP3. Results can record which source was retrieved, whether the expected evidence was found, whether the generated answer was supported, response time, and model usage. The same evaluation structure can then be reused as more RAG patterns are added.
