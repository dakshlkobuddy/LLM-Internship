import os
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import pickle

DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"
os.makedirs(INDEX_FOLDER, exist_ok=True)

# --- Embedding model ---
model_name = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=model_name)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- Extract text ---
def load_text(file_path):
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

docs, metadatas = [], []
for file in os.listdir(DATA_FOLDER):
    if file.endswith((".pdf", ".txt")):
        path = os.path.join(DATA_FOLDER, file)
        text = load_text(path)
        if not text:
            print(f"⚠️ Skipping {file}: no extractable text.")
            continue
        chunks = splitter.split_text(text)
        docs.extend(chunks)
        metadatas.extend([{"source": file}] * len(chunks))
        print(f"✅ {file}: {len(chunks)} chunks added.")

# --- Build and save FAISS index ---
db = FAISS.from_texts(docs, embedder, metadatas=metadatas)
db.save_local(INDEX_FOLDER)

print(f"\n✅ FAISS index built and saved in: {INDEX_FOLDER}")
