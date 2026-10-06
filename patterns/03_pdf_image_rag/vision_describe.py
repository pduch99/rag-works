import base64
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).parent

ROOT_DIR = (
    BASE_DIR /
    ".." /
    ".."
).resolve()

IMAGE_PATH = (
    BASE_DIR /
    "output" /
    "pages" /
    "page_1.png"
)

OUTPUT_PATH = (
    BASE_DIR /
    "output" /
    "vision_description.txt"
)

load_dotenv(
    ROOT_DIR /
    ".env"
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def encode_image(path):
    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


image_base64 = encode_image(
    IMAGE_PATH
)

prompt = """
Analyze this Metro State campus map.

Give a concise description of the visual
information needed for future question answering.

Focus on:

- building locations relative to each other
- streets
- parking areas
- tunnels and skyways
- important spatial relationships
- symbols or legend information that may matter

Use short bullet points.

Only describe information clearly visible
in the image.

Keep the response under 600 words.
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
                        "url": (
                            "data:image/png;base64,"
                            f"{image_base64}"
                        )
                    }
                }
            ]
        }
    ],
    max_completion_tokens=1500
)

description = (
    response
    .choices[0]
    .message
    .content
)

print(description)

OUTPUT_PATH.write_text(
    description,
    encoding="utf-8"
)

print(
    f"\nsaved to: {OUTPUT_PATH}"
)