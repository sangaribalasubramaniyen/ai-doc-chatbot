from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv("config.env")

# Load embeddings + vector DB + LLM ONCE when the server starts
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# Allow a frontend (browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The shape of the incoming request: { "query": "..." }
class Question(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "AI Document Q&A Chatbot API is running"}

@app.post("/ask")
def ask(question: Question):
    # RAG: retrieve → augment → generate
    docs = vectorstore.similarity_search(question.query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question.query}
Answer:"""
    response = llm.invoke(prompt)
    return {"answer": response.content}
