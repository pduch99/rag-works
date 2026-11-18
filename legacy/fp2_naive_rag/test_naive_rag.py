from naive_rag import retrieve

questions = [
    "What are the admission requirements?",
    "How do I register for classes?",
    "What resources are available for students?"
]

for question in questions:
    print(f"\nquestion: {question}")
    results = retrieve(question)
    for _, meta in results:
        print(f"- {meta['source']} chunk {meta['chunk']}")
