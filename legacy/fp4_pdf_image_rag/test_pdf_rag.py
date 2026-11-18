import json
from pathlib import Path

from pdf_rag import (
    index_data,
    text_only_answer,
    multimodal_answer
)

BASE = Path(__file__).resolve().parent
OUTPUT_PATH = BASE / "output" / "results.json"

questions = [
    "Where is the Student Center based on the map?",
    "Where is the Library and Learning Center located?",
    "Where is the parking ramp relative to the Student Center?",
    "Where is St. John's Hall relative to the Student Center?",
    "What building is south of E. 6th Street?",
    "What parking areas are near the Library?",
    "How would you get from the Student Center to the Library?",
    "What tunnels or skyways are shown on the map?"
]

index_data()

results = []

for i, question in enumerate(questions, start=1):
    print(f"\nQuestion {i}: {question}")

    text_answer, text_results = text_only_answer(
        question
    )

    image_answer, image_results = multimodal_answer(
        question
    )

    result = {
        "question": question,
        "text_only_retrieval": [
            {
                "source": meta["source"],
                "page": meta["page"],
                "type": meta["type"]
            }
            for _, meta in text_results
        ],
        "text_only_answer": text_answer,
        "multimodal_retrieval": [
            {
                "source": meta["source"],
                "page": meta["page"],
                "type": meta["type"]
            }
            for _, meta in image_results
        ],
        "multimodal_answer": image_answer
    }

    results.append(result)

    print("\nTEXT ONLY:")
    print(text_answer)

    print("\nMULTIMODAL:")
    print(image_answer)

OUTPUT_PATH.write_text(
    json.dumps(
        results,
        indent=2
    ),
    encoding="utf-8"
)

print(
    f"\nresults saved to: {OUTPUT_PATH}"
)