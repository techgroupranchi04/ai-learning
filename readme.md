# AI Learning Roadmap

[How Retrieval-Augmented Generation (RAG) Solves the Limitations of LLMs | by Vinod Rane | Medium](https://images.openai.com/static-rsc-4/nJy9CkmGLNTbezenhKafGMMcMGfG8jXh6Cxxmf0q7m6v_DZFd3Fftl2RDbGRd9Gf1UPV85UUnIuKEaA2qzcRI_qrQVORnjloTQ3QSHyHtpx27YpyuV5F-OSPp8Bt2sjelJfB2hi_l_CPbIdIdeizLr3Q6tA_NcUz56kz_KcyxA8?purpose=inline)

## 1. RAG — Retrieval-Augmented Generation

Start here

Learn how LLMs answer questions using external knowledge.

- Level 1: Documents, chunking, embeddings, vector databases, retrieval, and generation.
- Level 2: Hybrid search, reranking, multi-query retrieval, HyDE, and parent-document retrieval.
- Level 3: Knowledge graphs, GraphRAG, and entity relationships.
- Level 4: Agentic RAG, routing, adaptive retrieval, and corrective RAG (CRAG).

Project: Codebase RAG + Architecture Explorer

Resource: [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)

[The Autonomous Enterprise: A Comprehensive Guide to Building and Securing AI Agents - perfecXion.ai](https://images.openai.com/static-rsc-4/SjBpDh3MP9FDLZAXSQJR6WtmvOvqWp41j1yfvXwqjLW_b3FS4BXtI2Be2jlSKh_jpyTR9heBxgjg09e0naO58Rl0JwEht-p3IvkSbPbdcOYPwZtmlLnqFZQUPmcGiZRASGYMmBqoizJQjR7Q2ztEZjjiwVYAFf5fYk3DNveNgF4?purpose=inline)

## 2. AI Agents

Learn how an LLM chooses actions and uses tools to accomplish goals.

- Fundamentals: ReAct, tool calling, planning, memory, and reasoning loops.
- Frameworks: smolagents, LangGraph, LlamaIndex Workflows, and CrewAI.
- Multimodal AI: Vision-language models, audio, video, and document understanding.
- Multi-agent systems: Delegation, orchestration, and specialized agents.
- Evaluation: Test, debug, and certify your agent.

Project: AI Web Application Testing Agent

Resource: [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)

[Model Context Protocol - MCP Architecture Overview](https://images.openai.com/static-rsc-4/WE0qxUyWsPuLcUxiuoNjp_gQNqa2JOom_HD-SORUeWxIpsBHtJzq5dZ8DZoKVj-TkQeT9YrAgwVKpNDOsD38EWGCRPmwMIqlzc-jxAqbOTlGvoBWU00KR5sVUcWtw3Q_AOPuKWrEIhoUQb9XQ01Uq09wrJf_ZLfC99QGhXu6RwM?purpose=inline)

## 3. MCP — Model Context Protocol

Learn how to expose tools and data to compatible AI applications through a standard protocol.

- MCP clients and servers.
- Tools, resources, and prompts.
- Transports and server lifecycle.
- Authentication, permissions, and error handling.
- Integrating MCP with AI agents.

Project: MCP server for your own development or testing tools

Resources: [MCP course](https://huggingface.co/learn/mcp-course/) · [Official MCP documentation](https://modelcontextprotocol.io/)

[Fine-tune & Deploy Open-Source LLMs for Production](https://images.openai.com/static-rsc-4/rV5jLpeJ_aCxgIBEfTRm8EJHBUOPkZUeSGY5h5l1WohjBjzEtPWMae9AHQYz8TniVv3V1hZammKxCoHyc3PN4kIECXgXn8AIfwZi_yZtBVSpRdxyT4boy8zfbrEUaqlVL4hnoroyGYzBylWnZbkBp2XY_xuqXMBqhe3n-yXSpyA?purpose=inline)

## 4. Fine-Tuning LLMs

Learn how to adapt a pretrained model to a particular task or style.

- Supervised fine-tuning (SFT).
- LoRA and QLoRA.
- PEFT and parameter-efficient training.
- Dataset preparation and evaluation.
- Training, validation, and inference.

Project: Fine-tune a small model for structured QA test-case generation

Resources: [Hugging Face PEFT](https://huggingface.co/docs/peft) · [Fine-tuning course material](https://huggingface.co/learn/agents-course/bonus-unit1)

## The order I recommend

| Stage | Focus                         | What you should be able to build                                |
| ----- | ----------------------------- | --------------------------------------------------------------- |
| 1     | Basic RAG                     | Chat with your own documents                                    |
| 2     | Advanced RAG                  | More reliable retrieval and grounded answers                    |
| 3     | AI agent fundamentals         | An agent that uses tools                                        |
| 4     | Agent frameworks              | A structured, testable agent workflow                           |
| 5     | MCP                           | A reusable server exposing tools to AI clients                  |
| 6     | GraphRAG and Agentic RAG      | Knowledge-aware, adaptive retrieval systems                     |
| 7     | Multimodal and multi-agent AI | Systems that process different data types and coordinate agents |
| 8     | Fine-tuning                   | A model adapted to a specific task                              |

This order is a practical recommendation, not a strict prerequisite chain. You can learn MCP after understanding basic tool calling, for example, without waiting until you've completed every agent framework.

## How to study each topic

Use the same learning cycle for every topic:

### 01

Understand

Learn the concept, why it exists, and when to use it.

### 02

Implement

Build a small working example in Python.

### 03

Test

Use known inputs, expected outputs, and edge cases.

### 04

Integrate

Add the concept to a larger project.

### 05

Document

Record the architecture, trade-offs, failures, and lessons learned.

## Your first practical milestone

Begin with Basic RAG, using Python and a local or hosted LLM. Build a small application that:

1. Accepts `.txt` and PDF documents.
2. Splits documents into chunks.
3. Generates embeddings.
4. Retrieves the most relevant chunks for a question.
5. Generates an answer grounded in those chunks.
6. Shows the source documents or passages used in the answer.

Once that works, add retrieval evaluation before moving to hybrid search and reranking.
