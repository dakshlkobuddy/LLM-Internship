from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

app = FastAPI(title="AI Semantic Search Backend", version="1.0")

# --- CORS (optional if you use frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths
DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# --- Embedding Model & Splitter
MODEL_NAME = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- Helper: Extract text from file
def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        try:
            text = extract_text(file_path)
        except Exception:
            pdf = PdfReader(file_path)
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text.strip()

# --- Helper: Load FAISS (if exists)
def load_or_create_faiss():
    if os.path.exists(INDEX_FOLDER):
        try:
            db = FAISS.load_local(INDEX_FOLDER, embedder, allow_dangerous_deserialization=True)
            return db
        except:
            return FAISS.from_texts([], embedder)
    else:
        return FAISS.from_texts([], embedder)

# --- API 1: Upload multiple files
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    db = load_or_create_faiss()
    total_chunks = 0
    for file in files:
        file_path = os.path.join(DATA_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"✅ Saved {file.filename}")

        text = extract_text_from_file(file_path)
        if not text:
            print(f"⚠️ Skipping {file.filename}: No extractable text.")
            continue

        chunks = splitter.split_text(text)
        metadatas = [{"source": file.filename}] * len(chunks)
        db.add_texts(chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    db.save_local(INDEX_FOLDER)
    return {"status": "success", "files_processed": len(files), "chunks_added": total_chunks}

# --- API 2: Query search
@app.post("/query/")
async def search_query(query: str = Form(...), top_k: int = Form(5)):
    if not os.path.exists(INDEX_FOLDER):
        return {"error": "No index found. Please upload documents first."}

    db = FAISS.load_local(INDEX_FOLDER, embedder, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)

    if not results:
        return {"message": "No results found for your query."}

    response = []
    for r in results:
        response.append({
            "source": r.metadata.get("source", "Unknown"),
            "text": r.page_content
        })

    return {"query": query, "results": response}

# --- Root route
@app.get("/")
def root():
    return {"message": "Welcome to the Semantic Search API. Use /upload or /query endpoints."}
