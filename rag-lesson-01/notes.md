## Our learning path

| Phase          | What you'll learn                                         |
| -------------- | --------------------------------------------------------- |
| 1. RAG         | Basic RAG → advanced retrieval → GraphRAG → Agentic RAG   |
| 2. AI Agents   | Tool calling → ReAct → planning → frameworks → evaluation |
| 3. MCP         | Build servers, expose tools and resources, connect agents |
| 4. Fine-tuning | Datasets → SFT → LoRA → QLoRA → model evaluation          |

Each lesson will contain notes, diagrams where useful, practice code, expected output, exercises, and a checkpoint.

# Phase 1 — RAG Fundamentals

## Lesson 1: What is RAG, and why do we need it?

Level: Beginner · Time: 30–45 minutes · Language: Python

### 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain what RAG means.
- Understand why an LLM sometimes needs external information.
- Describe the three main steps of a RAG pipeline.
- Write a simple Python program that retrieves relevant information before answering a question.

## 2. What is RAG?

RAG stands for Retrieval-Augmented Generation.

It is a technique that combines information retrieval with an LLM to help generate answers using relevant external information.

Imagine you have a company document containing its leave policy. You ask an AI chatbot:

"How many casual leaves do employees get?"

The model may not know your company's specific policy. RAG allows the application to retrieve the relevant passage from that document and provide it to the model as context.

Step 1: User asks a question

"How many casual leaves are allowed?"

Step 2: Retrieval

Find the relevant passage in the documents.

Step 3: Generation

The LLM uses the question and retrieved passage to formulate an answer.

Answer

Employees are allowed 12 casual leave days per year.

Illustrative example: the answer is correct only if the document states this policy.

### 3. The three core concepts

[Folder with documents and cartoon hand holding magnifying glass. Files search. File manager, data storage and indexing.](https://images.openai.com/static-rsc-4/8sVDSmtdVu7iWJ7DFv4YEUzT069W34m_n81IDSfcqd-PhMyvYSWyG9ALrGtbchOhDPXbQpLgEU8jSDJ_j9HZr0wcBg0h_fVUEIaWD9daEAek-BgGeFn2tWJwfzYDxT6BMXxAyinLNlTNEM9fZcx2TVsf-XI_vI8P17VPwSll948kiEXYvbm8v04645Hf1DaA?purpose=inline)

A. Retrieval

Search a collection of documents and find information relevant to the user's question.

Example: Find the paragraph about casual leave.

[How Text Chunking Works: The Foundation of Every RAG System | by Saransh Surana | Medium](https://images.openai.com/static-rsc-4/i3VpRB2sHqK7G-4H1FGnIdiVTMQ9bso3-FxWw-wRJJQVPhS8hVt0lAIUutw161lOxsfgk-MdAEumkxQ8YUokAf8qJuitVjf7jFeS9OnE2Agw2h70ZhBME1ZYzF3l1V2nJ3EsNFz8405wS6LGsxL-M9QYHpmFy_sHUW12U_rXEKY?purpose=inline)

B. Augmentation

Add the retrieved information to the question as context for the LLM.

Example: Provide the leave policy paragraph alongside the question.

[Q&A With Your Docs: A Gentle Introduction to Matching Engine + PaLM | by John Grinalds | Google Cloud - Community | Medium](https://images.openai.com/static-rsc-4/eIRExkqb-ESl5CLlVYPcgXNMkOaKCuyXeCsaPzcHDLQRhM66DmL6F_nx66k12gmJRLLSH9VBHgbD1P5Tcwd9z78wX4SkWTDTQLdWVnRrfh0_S8dZ47M8LZz-lR1wDnwuuS6Pf-BnCs5P0IMNodCRr8U1K5iq1ob1YxfbVf9bFUA?purpose=inline)

C. Generation

Ask the LLM to formulate an answer using the supplied context.

Example: Explain the policy in a clear sentence.

### 4. Why do we need RAG?

| Without RAG                             | With RAG                                                             |
| --------------------------------------- | -------------------------------------------------------------------- |
| May not know private company data       | Can retrieve information from company documents                      |
| May use outdated knowledge              | Can retrieve updated information from the connected knowledge source |
| May invent unsupported details          | Can be instructed to answer from retrieved context                   |
| Often lacks document-level traceability | Can identify the passages used as sources                            |

Important: RAG does not guarantee correct answers. Retrieval can find the wrong passage, and the LLM can still misunderstand the context.

## 5. Practice code — build your first retrieval system

Before using LangChain, embeddings, or a vector database, let's build a tiny RAG-like pipeline using plain Python.

This is important because I want you to understand the logic before introducing libraries.

### Step 1: Create your project

Open VS Code or your terminal and create this structure:

```
rag-lesson-01/
└── main.py
```

You do not need to install any packages for this exercise.

### Step 2: Write the code

Copy the following code into `main.py`:

```
documents = [    "Python is a programming language used for web development and AI.",    "RAG stands for Retrieval-Augmented Generation.",    "Embeddings represent text as numerical vectors.",    "Chunking splits large documents into smaller pieces.",    "A vector database stores embeddings and supports similarity search."]question = "What does RAG stand for?"# Step 1: Retrieve a relevant document using keyword matchingquestion_words = set(question.lower().replace("?", "").split())best_document = Nonebest_score = 0for document in documents:    document_words = set(        document.lower().replace(".", "").split()    )    score = len(question_words & document_words)    if score > best_score:        best_score = score        best_document = document# Step 2: Augment the question with retrieved contextcontext = best_document if best_document else "No relevant document found."prompt = f"""Answer the question using the context below.Context: {context}Question: {question}Answer:"""# Step 3: Display the prompt that would be sent to an LLMprint("Question:", question)print("Retrieved document:", context)print("\nPrompt for the LLM:")print(prompt)
```

### Step 3: Run the program

In your terminal, navigate to the project folder and execute:

```
python main.py
```

If your system uses the `python3` command, run:

```
python3 main.py
```

### Step 4: Understand the expected output

The important output should look like this:

```
Question: What does RAG stand for?
Retrieved document: RAG stands for Retrieval-Augmented Generation.

Prompt for the LLM:

Answer the question using the context below.

Context: RAG stands for Retrieval-Augmented Generation.

Question: What does RAG stand for?

Answer:
```

Notice something important: this program has not generated an AI answer yet. It retrieves a document and constructs a prompt. In the next stage, we can connect an actual language model to generate the answer.

### 6. Understand the code

| Code concept     | Purpose                                          |
| ---------------- | ------------------------------------------------ |
| `documents`      | The knowledge available to our system            |
| `question`       | The user's query                                 |
| `question_words` | Words extracted from the query                   |
| `score`          | Number of matching words                         |
| `best_document`  | Document selected by the simple retrieval method |
| `context`        | Retrieved information supplied to the LLM        |
| `prompt`         | Instructions and context prepared for generation |

Our retriever uses keyword overlap. This is deliberately simple: it won't reliably recognize that "What is the meaning of RAG?" and "Define Retrieval-Augmented Generation" express similar ideas. Later, we'll learn how embeddings and semantic search help address this limitation.

## 7. Your practice exercises

Try these yourself before looking for solutions.

## Exercise checklist

Exercise 1: Change the question to `What are embeddings?` and run the program.

Exercise 2: Add a document explaining what LangChain is. Ask a question that should retrieve it.

Exercise 3: Add a question that has no matching document. Observe what happens and explain why this retriever is limited.

0 of 3 exercises completed

## 8. Checkpoint — test your understanding

Answer these three questions in your own words.

1\. What does RAG stand for?

Retrieval-Augmented Generation

Random AI Generation

Recursive Agent Gateway

2\. What is the purpose of retrieval?

Train a new LLM from scratch

Find relevant information for a question

Convert Python into JavaScript

3\. Does our current Python program generate an answer using an LLM?

Yes, it already uses an LLM.

No, it only retrieves text and prepares a prompt.

Check my answers

Your assignment: Run `main.py`, complete the three exercises, and answer the checkpoint. Send me your output or code if you encounter an error.

## Remember this distinction:

- Retrieval finds the information.
- Augmentation adds that information to the prompt.
- Generation uses an LLM to produce the answer.
