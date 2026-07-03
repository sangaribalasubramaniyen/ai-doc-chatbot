from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv("config.env")

# Load the same embedding model + the stored database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)

# Connect to the free Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

print("\n✅ Chatbot ready! Ask questions about your resume.\n")
while True:
    question = input("Ask (or type 'exit'): ")
    if question.lower() == "exit":
        break

# STEP 1 — RETRIEVE: find the 3 most relevant chunks from the resume
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # STEP 2 — AUGMENT: inject that context into the prompt
    prompt = f"""Answer the question using only the context below.

  Context:
  {context}

  Question: {question}
  Answer:"""

    # STEP 3 — GENERATE: the LLM answers from the context
    response = llm.invoke(prompt)
    print("Answer:", response.content, "\n")