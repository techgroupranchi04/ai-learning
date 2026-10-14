# Lesson 4 — Vector Stores and Retrieval

## 1. What problem does a vector store solve?

Imagine you have 1,000 document chunks. When a user asks a question, you need to find the most relevant chunks without manually comparing every result yourself.

A vector store helps you store text and its embeddings, then search for similar vectors efficiently.

Document

PDF, text file, or other source

Chunking + Embeddings

Split text and convert chunks into vectors

Vector Store

Stores text and searchable vector representations

Relevant chunks

Returned for the user's question

## 2. What we're going to build

We'll use LangChain's `InMemoryVectorStore` and the embedding model from Lesson 3.

The application will:

1. Create sample document chunks.
2. Generate and store their embeddings.
3. Accept a question from the user.
4. Search for the three most relevant chunks.
5. Display the retrieved text.

An in-memory store is ideal for learning. Its data disappears when the Python process ends, so later we'll explore persistent vector databases.

## 3. Set up the project

Create this folder structure:

```
rag-lesson-04/
├── main.py
└── requirements.txt
```

Add the following to `requirements.txt`:

```
langchain-core
langchain-community
langchain-huggingface
sentence-transformers
```

Install the packages inside your activated virtual environment:

```
pip install -r requirements.txt
```

If you already installed these packages in your previous RAG project, you can reuse that environment.

## 4. Build the retrieval application

Add this code to `main.py`:

```
from langchain_core.documents import Documentfrom langchain_core.vectorstores import InMemoryVectorStorefrom langchain_huggingface import HuggingFaceEmbeddings# 1. Load the embedding modelembeddings = HuggingFaceEmbeddings(    model_name="sentence-transformers/all-MiniLM-L6-v2")# 2. Prepare document chunksdocuments = [    Document(        page_content="Python is a programming language used for automation and AI.",        metadata={"topic": "Python"}    ),    Document(        page_content="RAG retrieves relevant information from external documents.",        metadata={"topic": "RAG"}    ),    Document(        page_content="Embeddings represent text as numerical vectors.",        metadata={"topic": "Embeddings"}    ),    Document(        page_content="Vector stores save embeddings and support similarity search.",        metadata={"topic": "Vector Stores"}    ),    Document(        page_content="Chunking divides large documents into smaller pieces.",        metadata={"topic": "Chunking"}    ),]# 3. Create the vector store and index the documentsvector_store = InMemoryVectorStore(embedding=embeddings)
```

## 5. Run your application

From the `rag-lesson-04` folder, execute:

```
python main.py
```

Try this question:

```
How can I search documents using their meaning?
```

Your output should begin with something like:

```
Indexed 5 document chunks.

Ask a question: How can I search documents using their meaning?

Retrieved chunks:

Result 1
Topic: ...
Text: ...
Similarity score: ...
```

The exact ordering and scores depend on the model and its similarity calculations. The results should generally favor chunks related to embeddings and vector search.

Important: `similarity_search_with_score()` returns a relevance distance/score according to the vector store's scoring convention. For this in-memory implementation, the returned score is a cosine-similarity-style distance, where a lower distance indicates a closer match. Do not interpret it as the same score convention as the cosine-similarity function you implemented in Lesson 3.

## 6. Understand the key lines

| Code                                       | Purpose                                            |
| ------------------------------------------ | -------------------------------------------------- |
| `Document(...)`                            | Stores text together with metadata.                |
| `InMemoryVectorStore(...)`                 | Creates an in-memory vector store.                 |
| `add_documents(documents)`                 | Embeds and indexes the documents.                  |
| `similarity_search_with_score(query, k=3)` | Retrieves up to three relevant chunks with scores. |
| `document.page_content`                    | Gets the retrieved text.                           |
| `document.metadata`                        | Gets additional information about the chunk.       |

One important distinction: retrieval is not generation. At this stage, your application finds relevant information but does not yet use an LLM to generate a natural-language answer.

## 7. Your practical checkpoint

After running the program, try these three questions:

- `What is Python used for?`
- `What are embeddings?`
- `How are large documents divided?`

Then answer these questions:

1. What does `k=3` mean?
2. Why do we store metadata alongside document text?
3. What happens to an `InMemoryVectorStore` when the Python process ends?
4. Does your current application generate an answer using an LLM, or only retrieve relevant chunks?

Share your output and answers. We'll debug any issues and make sure retrieval is clear before moving to the next lesson.
