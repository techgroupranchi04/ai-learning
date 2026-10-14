## Lesson 2: Text Preprocessing and Chunking

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain text preprocessing and why it matters.
- Split large documents into smaller chunks.
- Understand chunk size and chunk overlap.
- Implement character-based and word-based chunking in Python.
- Inspect the output and choose sensible chunk sizes for a document.

## 2. What is text preprocessing?

Text preprocessing means cleaning and preparing raw text before you store it or use it for retrieval.

Imagine a PDF contains this text:

```
   RAG   stands for Retrieval-Augmented Generation.


It helps LLMs answer questions using external documents.
```

The text has unnecessary spaces and blank lines. Preprocessing can make it more consistent:

```
RAG stands for Retrieval-Augmented Generation.

It helps LLMs answer questions using external documents.
```

Common preprocessing tasks include:

| Task                       | Purpose                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| Whitespace cleanup         | Remove unnecessary spaces and blank lines                          |
| Text extraction            | Extract text from PDFs, Word files, or HTML                        |
| Unicode normalization      | Standardize certain equivalent text representations                |
| Header and footer handling | Remove repeated page content when appropriate                      |
| Metadata preservation      | Keep document title, page number, source, and other useful details |

Important: Don't remove everything that looks unusual. Code indentation, table structure, punctuation, headings, and paragraph boundaries may carry useful meaning. Preprocessing should preserve information that helps answer questions.

## 3. What is chunking?

Chunking is the process of splitting a large document into smaller pieces of text called chunks.

Suppose your document contains 1,000 words about Python, RAG, embeddings, and vector databases. If a user asks about embeddings, searching the entire document as one large piece may be less useful than retrieving the specific section about embeddings.

Instead, we divide the document into chunks.

Original document

A long document containing many topics

Preprocessing and chunking

CHUNK 1

Python

Python is a programming language used for automation and AI.

CHUNK 2

RAG

RAG retrieves external information to help an LLM answer questions.

CHUNK 3

Embeddings

Embeddings represent text as numerical vectors.

CHUNK 4

Vector databases

Vector databases store vectors and support similarity search.

Illustrative example: each chunk represents one topic. Actual chunking doesn't always align perfectly with topic boundaries.

Why is this useful?

1. We can retrieve only the parts relevant to a question.
2. Smaller passages can fit more easily within an LLM's context window.
3. Embeddings can represent individual chunks instead of entire long documents.
4. We can associate each chunk with source metadata, such as a document name and page number.

## 4. Chunk size

Chunk size determines how much text goes into each chunk.

It might be measured in characters, words, or tokens, depending on the implementation.

For example, suppose we use character-based chunking with a chunk size of 100.

- Chunk 1 contains up to 100 characters.
- Chunk 2 contains the next group of up to 100 characters.
- The process continues until the text is covered.

A larger chunk may preserve more context, while a smaller chunk may offer more precise retrieval. Neither is always better.

| Chunk size | Potential benefit                     | Potential drawback                                  |
| ---------- | ------------------------------------- | --------------------------------------------------- |
| Small      | More focused passages                 | Can lose important context                          |
| Medium     | Balance between context and precision | May split a topic                                   |
| Large      | More surrounding context              | May retrieve irrelevant text along with useful text |

There is no universally ideal chunk size. The best value depends on document structure, retrieval method, embedding model, and question type.

## 5. Chunk overlap

Chunk overlap is the amount of text shared between neighboring chunks.

Why do we need it? Because important information can fall across a chunk boundary.

Consider this sentence:

> A vector database stores embeddings. These embeddings represent text as numerical vectors.

If the document is split exactly between those two sentences, the second chunk might lose the context that explains what "These embeddings" refers to.

Overlap preserves some of the previous chunk's text in the next one.

Example: 10-word chunks with 3-word overlap

CHUNK 1

Python is useful for building AI applications and automating tasks.

CHUNK 2

AI applications and automating tasks with Python helps developers work faster.

The shared words preserve continuity. We'll verify the exact overlap behavior in the code below.

For a fixed-size sliding window, the relationship is:

\\[ \text{Step size} = \text{Chunk size} - \text{Overlap} \\]

For example, if the chunk size is 100 characters and overlap is 20 characters, the next chunk starts 80 characters after the previous chunk's start.

- Chunk size: 100
- Overlap: 20
- Step size: 80

Overlap is useful, but too much overlap creates redundant chunks, increases storage requirements, and can cause near-duplicate retrieval results.

## 6. Practical coding — build a chunking program

Let's start with plain Python, without LangChain. This will help you understand what a chunking library does behind the scenes.

### Step 1: Create the project

Create a new folder:

```
rag-lesson-02/
├── main.py
└── sample_document.txt
```

Put this content in `sample_document.txt`:

```
Python is a programming language used for automation and AI.
RAG stands for Retrieval-Augmented Generation.
RAG helps language models answer questions using external documents.
Embeddings represent text as numerical vectors.
Vector databases store embeddings and support similarity search.
Chunking divides large documents into smaller pieces.
```

### Step 2: Clean the text and split it into chunks

Add this code to `main.py`.

```
from pathlib import Pathdef clean_text(text):    """Remove extra whitespace and blank lines."""    lines = [line.strip() for line in text.splitlines()]    lines = [line for line in lines if line]    return "\n".join(lines)def chunk_text(text, chunk_size=100, overlap=20):    """Split text into overlapping character-based chunks."""    if chunk_size <= 0:        raise ValueError("chunk_size must be greater than zero")    if overlap < 0 or overlap >= chunk_size:        raise ValueError("overlap must be >= 0 and < chunk_size")    chunks = []    step = chunk_size - overlap    for start in range(0, len(text), step):        chunk = text[start:start + chunk_size]        chunks.append(chunk)    return chunks# 1. Read the documentraw_text = Path("sample_document.txt").read_text(encoding="utf-8")# 2. Preprocess the documentcleaned_text = clean_text(raw_text)# 3. Split the text into chunkschunks = chunk_text(cleaned_text, chunk_size=100, overlap=20)# 4. Display the resultsprint("CLEANED DOCUMENT:")print(cleaned_text)print("\nTOTAL CHARACTERS:", len(cleaned_text))print("TOTAL CHUNKS:", len(chunks))for index, chunk in enumerate(chunks, start=1):    print(f"\n--- Chunk {index} ({len(chunk)} characters) ---")    print(chunk)
```

### Step 3: Run the code

Open the terminal in your project folder and execute:

```
python main.py
```

On systems where Python uses the `python3` command:

```
python3 main.py
```

### Step 4: Understand the output

Your output should show:

- The cleaned document.
- The total number of characters.
- The total number of chunks.
- Each chunk's text and character count.

With the supplied six-line sample, the cleaned text contains 334 characters. With a chunk size of 100 and overlap of 20, you should get five chunks.

The chunks overlap because each new chunk starts 80 characters after the previous chunk's starting position.

Important coding detail: Our final chunk may be shorter than 100 characters. That's expected. Also, this character-based implementation may split words or sentences in the middle. We'll improve the chunking strategy after you understand the basics.

## 7. Experiment with chunk size and overlap

Change only the arguments in this line:

```
chunks = chunk_text(cleaned_text, chunk_size=100, overlap=20)
```

Try these configurations:

| Experiment | Chunk size | Overlap |
| ---------- | ---------- | ------- |
| A          | 50         | 0       |
| B          | 50         | 10      |
| C          | 100        | 20      |
| D          | 200        | 40      |

Before running each experiment, predict how many chunks you expect. Then run the program and compare your prediction with the actual result.

One correction to the expected output above: with the exact sample text provided, the cleaned document contains 343 characters, not 334. The expected total remains 5 chunks for a chunk size of 100 and overlap of 20.

## 8. Other chunking strategies

Our current code splits text by character positions. In real RAG applications, you may encounter these strategies:

Fixed-size chunking

Splits text according to a fixed character or token count. Simple, but can split sentences or ideas.

Sentence-based chunking

Splits text into sentences and groups them into chunks. Helps preserve sentence boundaries.

Recursive character chunking

Tries to split at preferred boundaries, such as paragraphs, then lines, then spaces, before falling back to smaller units. LangChain's `RecursiveCharacterTextSplitter` is a common implementation.

Semantic chunking

Uses meaning or semantic similarity to determine where related content should stay together. It can preserve topic coherence, but adds complexity and may require embeddings.

We will implement recursive character chunking with LangChain in a later exercise. For now, make sure you understand how the basic version works.

## 9. Practice assignment

Complete these tasks in your `rag-lesson-02` project.

Task 1: Run the program with chunk size 50 and overlap 0.

Task 2: Set overlap to 10 and observe what text is repeated between chunks.

Task 3: Modify `clean_text()` to convert repeated spaces within a line into single spaces.

Task 4: Try overlap equal to chunk size and explain why the program rejects it.

0 of 4 tasks completed

## 10. Checkpoint quiz

1\. Why do we split large documents into chunks?

To reduce the number of documents permanently

To retrieve smaller, relevant passages and manage context

To train the LLM from scratch

2\. If chunk size is 100 and overlap is 20, what is the step size?

20 characters

80 characters

120 characters

3\. What can happen when the overlap is too large?

More redundant text and storage

All embeddings disappear

The LLM automatically becomes more accurate

4\. Which chunking strategy tries preferred boundaries such as paragraphs and spaces?

Random chunking

Recursive character chunking

Fine-tuning

Mentor's instruction: Finish the practice tasks and checkpoint before moving on. Our next lesson will cover embeddings and vector similarity, where you'll learn how a RAG system finds relevant content even when the user's wording differs from the document.
