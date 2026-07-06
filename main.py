from dotenv import load_dotenv
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import shutil

load_dotenv("config.env")

# Load embeddings + vector DB + LLM ONCE when the server starts
embeddings = FastEmbedEmbeddings()
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
    return FileResponse("index.html")

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

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vectorstore  # so we can replace the document the bot knows about
    # 1. Save the uploaded PDF to disk
    with open("uploaded.pdf", "wb") as f:
        shutil.copyfileobj(file.file, f)
    # 2. Load the PDF and split it into chunks
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    # 3. Rebuild the vector store from the NEW document (in memory)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return {"message": f"Uploaded '{file.filename}' - {len(chunks)} sections indexed. Ask away!"}
