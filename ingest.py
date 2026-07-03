from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

load_dotenv("config.env")

# 1. Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Create embeddings (free, lightweight - uses ONNX, not PyTorch)
print("Loading embedding model...")
embeddings = FastEmbedEmbeddings()

# 4. Store in Chroma vector database
print("Storing in vector database...")
Chroma.from_documents(chunks, embeddings, persist_directory="./db")
print("✅ Done! PDF is now stored in the vector database.")