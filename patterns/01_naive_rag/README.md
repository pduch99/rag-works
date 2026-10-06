# Naive RAG

Naive RAG is the basic RAG pattern. A question is used to search a vector database for relevant information, then the retrieved information is passed to an LLM to generate an answer.

## Flow

Question -> Vector Search -> Top Documents -> LLM -> Answer

## Metro State Example

This implementation uses Metro State documentation as the knowledge base. The documents are split into chunks and stored in ChromaDB. When a question is asked, the most similar chunks are retrieved and passed to the LLM as context.

## Run

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add an OpenAI API key.

Add `.txt` files to `data/metrostate_docs/`, then run this from the `01_naive_rag` folder:

```bash
python naive_rag.py
```

## Files

- `naive_rag.py` - naive RAG implementation
- `test_naive_rag.py` - example questions for testing retrieval
- `data/metrostate_docs/` - input documents

## Current Limitations

- only supports text files
- fixed chunk size
- vector search only
- no reranking
- no query rewriting
- one retrieval step

These limitations are intentional because this implementation is meant to demonstrate the basic Naive RAG pattern.
