
documents = [
    "Python is a programming language used for web development and AI.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Embeddings represent text as numerical vectors.",
    "Chunking splits large documents into smaller pieces.",
    "A vector database stores embeddings and supports similarity search."
]

question = "What are embeddings?"

# Step 1: Retrieve a relevant document using keyword matching
question_words = set(question.lower().replace("?", "").split())

best_document = None
best_score = 0

for document in documents:
    document_words = set(
        document.lower().replace(".", "").split()
    )

    score = len(question_words & document_words)

    if score > best_score:
        best_score = score
        best_document = document

# Step 2: Augment the question with retrieved context
context = best_document if best_document else "No relevant document found."

prompt = f"""
Answer the question using the context below.

Context: {context}

Question: {question}

Answer:
"""

# Step 3: Display the prompt that would be sent to an LLM
print("Question:", question)
print("Retrieved document:", context)
print("\nPrompt for the LLM:")
print(prompt)
