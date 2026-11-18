from pathlib import Path
import base64
import os

import chromadb
import pymupdf
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent

PDF_PATH = BASE / "data" / "source.pdf"
OUTPUT_DIR = BASE / "output"
PAGES_DIR = OUTPUT_DIR / "pages"
REGIONS_DIR = OUTPUT_DIR / "regions"
DESCRIPTIONS_PATH = OUTPUT_DIR / "region_descriptions.txt"

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

PAGES_DIR.mkdir(parents=True, exist_ok=True)
REGIONS_DIR.mkdir(parents=True, exist_ok=True)


def encode_image(path):
    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


def describe_region(path, page_num, region_name):
    image_base64 = encode_image(path)

    prompt = f"""
Analyze this cropped region from page {page_num}
of a Metro State PDF.

Region: {region_name}

Describe only information clearly visible in this crop.

Focus on:
- building names
- roads
- parking
- spatial relationships
- tunnels or skyways
- other useful visual details

Keep it concise.
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
        max_completion_tokens=500
    )

    return response.choices[0].message.content


def split_regions(image_path, page_num):
    image = Image.open(image_path)

    width, height = image.size

    mid_x = width // 2
    mid_y = height // 2

    regions = {
        "nw": (0, 0, mid_x, mid_y),
        "ne": (mid_x, 0, width, mid_y),
        "sw": (0, mid_y, mid_x, height),
        "se": (mid_x, mid_y, width, height)
    }

    saved = []

    for region_name, box in regions.items():
        crop = image.crop(box)

        path = (
            REGIONS_DIR /
            f"page_{page_num}_{region_name}.png"
        )

        crop.save(path)

        saved.append(
            (region_name, path)
        )

    return saved


def index_text(page_num, text):
    collection.upsert(
        ids=[
            f"page-{page_num}-text"
        ],
        documents=[
            text
        ],
        metadatas=[
            {
                "source": "source.pdf",
                "page": page_num,
                "type": "text"
            }
        ]
    )


def index_region(
    page_num,
    region_name,
    description,
    image_path
):
    collection.upsert(
        ids=[
            f"page-{page_num}-{region_name}"
        ],
        documents=[
            description
        ],
        metadatas=[
            {
                "source": "source.pdf",
                "page": page_num,
                "type": "visual_region",
                "region": region_name,
                "path": str(
                    image_path.relative_to(BASE)
                )
            }
        ]
    )


doc = pymupdf.open(PDF_PATH)

all_descriptions = []

print(f"PDF: {PDF_PATH.name}")
print(f"Pages: {len(doc)}")

for page_num, page in enumerate(
    doc,
    start=1
):
    print(
        f"\nProcessing page {page_num}..."
    )

    text = page.get_text()

    index_text(
        page_num,
        text
    )

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2)
    )

    page_path = (
        PAGES_DIR /
        f"page_{page_num}.png"
    )

    pix.save(page_path)

    print(
        f"saved page image: {page_path.name}"
    )

    regions = split_regions(
        page_path,
        page_num
    )

    for region_name, region_path in regions:
        print(
            f"describing page {page_num} "
            f"region {region_name}..."
        )

        description = describe_region(
            region_path,
            page_num,
            region_name
        )

        index_region(
            page_num,
            region_name,
            description,
            region_path
        )

        all_descriptions.append(
            f"""
PAGE {page_num}
REGION {region_name}

{description}
"""
        )

        print(
            f"indexed {region_path.name}"
        )

doc.close()

DESCRIPTIONS_PATH.write_text(
    "\n".join(all_descriptions),
    encoding="utf-8"
)

print("\nDone")
print(
    f"descriptions saved to: "
    f"{DESCRIPTIONS_PATH}"
)