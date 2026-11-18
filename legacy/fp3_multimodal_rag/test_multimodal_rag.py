from multimodal_rag import retrieve

questions = [
    "When do Fall 2026 classes begin?",
    "How do I register for classes?",
    "What student support resources are available?",
    "What steps are shown in the Naive RAG architecture?",
    "What does the final RAG Works interface show for Advanced RAG?",
    "What is the long-term goal shown in the RAG Works visual?"
]

for question in questions:
    print(f"\nquestion: {question}")
    results = retrieve(question)
    for _, meta in results:
        extra = f" chunk {meta['chunk']}" if "chunk" in meta else ""
        print(f"- [{meta['type']}] {meta['source']}{extra}")
