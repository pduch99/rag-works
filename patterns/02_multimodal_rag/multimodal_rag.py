import base64
import mimetypes
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb

BASE = Path(__file__).resolve().parent

load_dotenv(BASE.parent.parent / ".env")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chroma = chromadb.PersistentClient(
    path=str(BASE / "chroma_db")
)

collection = chroma.get_or_create_collection(
    "metrostate_multimodal"
)


def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0]

    if not mime:
        mime = "image/png"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


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

    text_context = []
    images = []

    for doc, meta in results:
        text_context.append(
            f"Source: {meta['source']} "
            f"({meta['type']})\n{doc}"
        )

        if (
            meta["type"] == "image"
            and meta.get("path")
        ):
            path = BASE / meta["path"]

            if path.exists():
                images.append(path)

    context = "\n\n".join(text_context)

    prompt = f"""
Answer using only the retrieved sources below
and any retrieved images attached to this message.

If the answer is not supported by the sources,
say you could not find it.

Retrieved sources:

{context}

Question:
{question}
"""

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    for path in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(path)
                }
            }
        )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_completion_tokens=1200
    )

    return (
        response.choices[0].message.content,
        results
    )


if __name__ == "__main__":
    print(
        "Multimodal RAG - type exit to quit"
    )

    while True:
        question = input(
            "\nquestion: "
        ).strip()

        if question.lower() in {
            "quit",
            "exit"
        }:
            break

        answer, results = ask(question)

        print("\nretrieved:")

        for _, meta in results:
            extra = ""

            if "chunk" in meta:
                extra = (
                    f" chunk {meta['chunk']}"
                )

            print(
                f"- [{meta['type']}] "
                f"{meta['source']}{extra}"
            )

        print(
            f"\nanswer:\n{answer}"
        )