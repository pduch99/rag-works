import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb

BASE = Path(__file__).resolve().parent

load_dotenv(BASE.parent.parent / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma = chromadb.PersistentClient(
    path=str(BASE / "chroma_db")
)

collection = chroma.get_or_create_collection("metrostate")


def load_docs(folder):
    docs = []

    for file in Path(folder).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        docs.append((file.name, text))

    return docs


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap

    return chunks


def index_docs(folder):
    docs = load_docs(folder)

    ids = []
    texts = []
    metadata = []

    for filename, text in docs:
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}-{i}")
            texts.append(chunk)
            metadata.append({
                "source": filename,
                "chunk": i
            })

    if ids:
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadata
        )

    print(f"indexed {len(ids)} chunks")


def retrieve(question, n=3):
    results = collection.query(
        query_texts=[question],
        n_results=n
    )

    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    return list(zip(docs, metadata))


def ask(question):
    results = retrieve(question)

    context = "\n\n".join(
        doc for doc, _ in results
    )

    prompt = f"""
Answer the question using only the provided Metro State information.

If the answer is not in the context, say you could not find it.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content, results


if __name__ == "__main__":
    index_docs(BASE / "data" / "metrostate_docs")

    while True:
        question = input("\nquestion: ").strip()

        if question.lower() in ["quit", "exit"]:
            break

        answer, results = ask(question)

        print("\nretrieved:")

        for _, meta in results:
            print(
                f"- {meta['source']} "
                f"chunk {meta['chunk']}"
            )

        print(f"\nanswer:\n{answer}")