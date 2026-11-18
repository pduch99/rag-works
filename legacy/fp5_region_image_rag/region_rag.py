import base64
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent

load_dotenv(ROOT / ".env")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chroma = chromadb.PersistentClient(
    path=str(BASE / "chroma_db")
)

collection = chroma.get_or_create_collection(
    "region_image_rag"
)


def encode_image(path):
    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


def retrieve(question, n=5):
    results = collection.query(
        query_texts=[question],
        n_results=n,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return list(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    )


def ask(question):
    results = retrieve(question)

    visual_results = []
    text_results = []

    for doc, meta, distance in results:
        if meta["type"] == "visual_region":
            visual_results.append(
                (doc, meta, distance)
            )
        else:
            text_results.append(
                (doc, meta, distance)
            )

    # Keep only the best 2 visual regions.
    selected_visuals = visual_results[:2]

    # Keep only the best text result.
    selected_text = text_results[:1]

    selected_results = (
        selected_visuals +
        selected_text
    )

    context = []
    images = []

    for doc, meta, distance in selected_results:
        region = meta.get("region", "-")

        context.append(
            f"""
Source: {meta['source']}
Page: {meta['page']}
Type: {meta['type']}
Region: {region}

{doc}
"""
        )

        if (
            meta["type"] == "visual_region"
            and meta.get("path")
        ):
            image_path = BASE / meta["path"]

            if image_path.exists():
                images.append(
                    (
                        region,
                        image_path,
                        distance
                    )
                )

    prompt = f"""
Answer the question using only the retrieved
PDF evidence below.

Use the cropped visual regions to understand
spatial relationships.

If the evidence is not enough to answer
confidently, say so.

Retrieved evidence:

{''.join(context)}

Question:
{question}
"""

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    for region, path, distance in images:
        image_base64 = encode_image(path)

        content.append(
            {
                "type": "text",
                "text": (
                    f"Visual region: {region}"
                )
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64,"
                        f"{image_base64}"
                    )
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
        max_completion_tokens=600
    )

    return (
        response.choices[0].message.content,
        results,
        selected_results
    )


if __name__ == "__main__":
    print("\nRegion-Level PDF RAG")
    print("type exit to quit")

    while True:
        question = input(
            "\nquestion: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:
            break

        answer, results, selected = ask(
            question
        )

        print("\nall retrieved candidates:")

        for _, meta, distance in results:
            print(
                f"- {meta['source']} "
                f"| page {meta['page']} "
                f"| {meta['type']} "
                f"| region "
                f"{meta.get('region', '-')} "
                f"| distance {distance:.4f}"
            )

        print("\nselected for Groq:")

        for _, meta, distance in selected:
            print(
                f"- {meta['type']} "
                f"| region "
                f"{meta.get('region', '-')} "
                f"| distance {distance:.4f}"
            )

        print(
            f"\nanswer:\n{answer}"
        )