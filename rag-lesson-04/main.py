
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Prepare document chunks
documents = [
    Document(
        page_content="Python is a programming language used for automation and AI.",
        metadata={"topic": "Python"}
    ),
    Document(
        page_content="RAG retrieves relevant information from external documents.",
        metadata={"topic": "RAG"}
    ),
    Document(
        page_content="Embeddings represent text as numerical vectors.",
        metadata={"topic": "Embeddings"}
    ),
    Document(
        page_content="Vector stores save embeddings and support similarity search.",
        metadata={"topic": "Vector Stores"}
    ),
    Document(
        page_content="Chunking divides large documents into smaller pieces.",
        metadata={"topic": "Chunking"}
    ),
]

# 3. Create the vector store and index the documents
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(documents)

print(f"Indexed {len(documents)} document chunks.")

# 4. Ask the user a question
query = input("\nAsk a question: ")

# 5. Retrieve the three most similar chunks
results = vector_store.similarity_search_with_score(
    query,
    k=3
)

# 6. Display the results
print("\nRetrieved chunks:")

for index, (document, score) in enumerate(results, start=1):
    print(f"\nResult {index}")
    print(f"Topic: {document.metadata['topic']}")
    print(f"Text: {document.page_content}")
    print(f"Similarity score: {score:.4f}")
