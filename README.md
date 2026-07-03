# 📄 AI Document Q&A Chatbot (RAG)

An AI chatbot that answers questions about a PDF using Retrieval-Augmented
Generation (RAG). Built with Python, FastAPI, and LangChain.

## Tech Stack
- Backend: Python, FastAPI
- AI/RAG: LangChain, Groq (LLaMA 3.1), HuggingFace embeddings
- Vector DB: Chroma
- Frontend: HTML, JavaScript

## How RAG works here
1. Ingest: PDF → chunks → embeddings → Chroma
2. Retrieve top 3 relevant chunks for the question
3. Augment: inject chunks into the prompt
4. Generate: LLM answers from that context (prevents hallucination)

## Author
Sangari B