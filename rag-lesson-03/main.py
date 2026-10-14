
from langchain_huggingface import HuggingFaceEmbeddings

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Our document chunks
sentences = [
    "Python is useful for automation.",
    "Python helps automate repetitive tasks.",
    "I enjoy watching movies."
]

# Convert document chunks into vectors
vectors = embeddings.embed_documents(sentences)

# Display the results
for index, vector in enumerate(vectors):
    print(f"Sentence {index + 1}: {sentences[index]}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print()

# uncommnet to compare similarity between sentences

# import math

# # User's question
# query = "How can I automate tasks?"

# # Convert the question into an embedding
# query_vector = embeddings.embed_query(query)


# def cosine_similarity(vector_a, vector_b):
#     dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

#     magnitude_a = math.sqrt(sum(a * a for a in vector_a))
#     magnitude_b = math.sqrt(sum(b * b for b in vector_b))

#     if magnitude_a == 0 or magnitude_b == 0:
#         return 0.0

#     return dot_product / (magnitude_a * magnitude_b)


# # Compare the query with every sentence
# results = []

# for sentence, vector in zip(sentences, vectors):
#     score = cosine_similarity(query_vector, vector)
#     results.append((sentence, score))

# # Sort from highest similarity to lowest
# results.sort(key=lambda item: item[1], reverse=True)

# print(f"\nQuery: {query}")
# print("\nSimilarity results:")

# for sentence, score in results:
#     print(f"{score:.4f} - {sentence}")