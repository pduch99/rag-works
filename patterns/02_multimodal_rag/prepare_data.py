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


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(
            text[start:start + size]
        )

        start += size - overlap

    return chunks


def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0]

    if not mime:
        mime = "image/png"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


def describe_image(path):
    prompt = """
Describe this image for retrieval.

Focus on:
- important headings and labels
- architecture steps
- relationships between items
- comparisons
- important visual information

Be factual and concise.
"""

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
                            "url": image_data_url(path)
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=1200
    )

    return response.choices[0].message.content


def index_text():
    count = 0

    folder = (
        BASE /
        "data" /
        "metrostate_docs"
    )

    for file in folder.glob("*.txt"):
        text = file.read_text(
            encoding="utf-8"
        )

        for i, chunk in enumerate(
            chunk_text(text)
        ):
            collection.upsert(
                ids=[
                    f"text-{file.stem}-{i}"
                ],
                documents=[chunk],
                metadatas=[
                    {
                        "source": file.name,
                        "type": "text",
                        "chunk": i
                    }
                ]
            )

            count += 1

    return count


def index_images():
    count = 0

    folder = BASE / "data" / "images"

    for file in folder.glob("*"):
        if file.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        }:
            continue

        print(
            f"\ndescribing {file.name}..."
        )

        description = describe_image(file)

        collection.upsert(
            ids=[
                f"image-{file.stem}"
            ],
            documents=[
                description
            ],
            metadatas=[
                {
                    "source": file.name,
                    "type": "image",
                    "path": str(
                        file.relative_to(BASE)
                    )
                }
            ]
        )

        print(
            f"\n{file.name} description:"
        )
        print(description)

        count += 1

    return count


if __name__ == "__main__":
    text_count = index_text()
    image_count = index_images()

    print(
        f"\nindexed {text_count} text chunks "
        f"and {image_count} images"
    )