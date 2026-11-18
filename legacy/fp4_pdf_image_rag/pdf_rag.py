import base64
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent

TEXT_PATH = BASE / "output" / "pdf_text.txt"
VISION_PATH = BASE / "output" / "vision_description.txt"
IMAGE_PATH = BASE / "output" / "pages" / "page_1.png"

load_dotenv(ROOT / ".env")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chroma = chromadb.PersistentClient(
    path=str(BASE / "chroma_db")
)

collection = chroma.get_or_create_collection(
    "pdf_image_rag"
)


def encode_image(path):
    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


def index_data():
    text = TEXT_PATH.read_text(
        encoding="utf-8"
    )

    vision = VISION_PATH.read_text(
        encoding="utf-8"
    )

    collection.upsert(
        ids=[
            "pdf-text-page-1",
            "pdf-visual-page-1"
        ],
        documents=[
            text,
            vision
        ],
        metadatas=[
            {
                "source": "source.pdf",
                "page": 1,
                "type": "text"
            },
            {
                "source": "source.pdf",
                "page": 1,
                "type": "visual"
            }
        ]
    )

    print("indexed PDF text and visual description")


def retrieve_text(question):
    results = collection.query(
        query_texts=[question],
        n_results=1,
        where={
            "type": "text"
        }
    )

    return list(
        zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    )


def retrieve_multimodal(question):
    results = collection.query(
        query_texts=[question],
        n_results=2
    )

    return list(
        zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    )


def text_only_answer(question):
    results = retrieve_text(question)

    context = "\n\n".join(
        doc for doc, _ in results
    )

    prompt = f"""
Answer the question using only the extracted
text from the PDF.

Do not use outside knowledge.

If the extracted text does not contain enough
information, say you could not determine it
from the text.

PDF text:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=700
    )

    return (
        response.choices[0].message.content,
        results
    )


def multimodal_answer(question):
    results = retrieve_multimodal(question)

    context = []

    for doc, meta in results:
        context.append(
            f"""
Source: {meta['source']}
Page: {meta['page']}
Type: {meta['type']}

{doc}
"""
        )

    prompt = f"""
Answer the question using the retrieved PDF
information and the actual PDF page image.

Pay close attention to spatial relationships
shown visually on the map.

Do not use outside knowledge.

If you cannot determine the answer clearly,
say so.

Retrieved evidence:

{''.join(context)}

Question:
{question}
"""

    image_base64 = encode_image(
        IMAGE_PATH
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                f"{image_base64}"
                            )
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=700
    )

    return (
        response.choices[0].message.content,
        results
    )


if __name__ == "__main__":
    index_data()

    print("\nPDF Image RAG Comparison")
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

        text_answer, text_results = (
            text_only_answer(question)
        )

        image_answer, image_results = (
            multimodal_answer(question)
        )

        print("\n-------------------------")
        print("TEXT-ONLY RETRIEVAL")
        print("-------------------------")

        for _, meta in text_results:
            print(
                f"{meta['source']} "
                f"| page {meta['page']} "
                f"| {meta['type']}"
            )

        print("\nTEXT-ONLY ANSWER:")
        print(text_answer)

        print("\n-------------------------")
        print("MULTIMODAL RETRIEVAL")
        print("-------------------------")

        for _, meta in image_results:
            print(
                f"{meta['source']} "
                f"| page {meta['page']} "
                f"| {meta['type']}"
            )

        print("\noriginal PDF image sent: True")

        print("\nMULTIMODAL ANSWER:")
        print(image_answer)