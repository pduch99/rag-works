from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).parent

PDF_PATH = (
    BASE_DIR /
    "data" /
    "source.pdf"
)

OUTPUT_DIR = (
    BASE_DIR /
    "output"
)

PAGES_DIR = (
    OUTPUT_DIR /
    "pages"
)

TEXT_PATH = (
    OUTPUT_DIR /
    "pdf_text.txt"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

PAGES_DIR.mkdir(
    exist_ok=True
)

doc = pymupdf.open(
    PDF_PATH
)

all_text = []

print(
    f"PDF: {PDF_PATH.name}"
)

print(
    f"Pages: {len(doc)}"
)

for page_num, page in enumerate(
    doc,
    start=1
):
    print(
        f"\nProcessing page {page_num}..."
    )

    text = page.get_text()

    all_text.append(
        f"\n--- PAGE {page_num} ---\n{text}"
    )

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2)
    )

    image_path = (
        PAGES_DIR /
        f"page_{page_num}.png"
    )

    pix.save(image_path)

    print(
        f"text characters: {len(text)}"
    )

    print(
        f"saved image: {image_path.name}"
    )

TEXT_PATH.write_text(
    "\n".join(all_text),
    encoding="utf-8"
)

doc.close()

print("\nDone")

print(
    f"text saved to: {TEXT_PATH}"
)

print(
    f"page images saved to: {PAGES_DIR}"
)