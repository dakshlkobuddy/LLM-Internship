# ------------------------------------------------------
# app.py — Streamlit Frontend for Gemini + FAISS Search
# ------------------------------------------------------

import streamlit as st
import requests
import os

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Semantic Search (Gemini + FAISS)",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# 🎨 Header
# ----------------------------
st.title("🧠 AI Document Search using Gemini + FAISS")
st.write("Upload PDFs and ask questions — the system will find relevant answers using semantic search powered by **Gemini Embeddings**.")

# ----------------------------
# 📁 Upload Section
# ----------------------------
st.header("📤 Upload Your PDF/Text Files")
uploaded_files = st.file_uploader(
    "Select multiple files (PDF or TXT):",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Upload to Knowledge Base"):
    if uploaded_files:
        with st.spinner("Uploading and processing files... ⏳"):
            files = [("files", (f.name, f, f.type)) for f in uploaded_files]
            response = requests.post(f"{API_URL}/upload/", files=files)
            if response.status_code == 200:
                res = response.json()
                st.success(f"✅ Uploaded {res['files_processed']} files and added {res['chunks_added']} chunks.")
            else:
                st.error(f"❌ Upload failed: {response.text}")
    else:
        st.warning("Please select at least one file to upload.")


# ----------------------------
# 🔍 Query Section
# ----------------------------
st.header("💬 Ask a Question About Your Documents")

query = st.text_input("Enter your question:")
top_k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if query.strip():
        with st.spinner("Searching for relevant answers... 🔍"):
            data = {"query": query, "top_k": top_k}
            response = requests.post(f"{API_URL}/query/", data=data)

            if response.status_code == 200:
                res = response.json()

                if "error" in res:
                    st.error(res["error"])
                elif "results" in res:
                    st.success("✅ Search completed successfully!")

                    # Optional summary (if you add summarization in backend later)
                    if "summary" in res:
                        st.subheader("🧠 Gemini Summary:")
                        st.info(res["summary"])

                    st.subheader("📄 Top Matching Results:")
                    for i, result in enumerate(res["results"], start=1):
                        st.markdown(f"### 🔹 Result {i}")
                        st.markdown(f"**Source File:** {result.get('source', 'Unknown')}")
                        st.markdown(f"**Text Snippet:** {result['text']}")
                        st.markdown("---")
                else:
                    st.warning("No results found for your query.")
            else:
                st.error(f"❌ Error: {response.status_code}")
    else:
        st.warning("Please enter a query first.")


# ----------------------------
# ⚙️ Backend Status
# ----------------------------
st.sidebar.header("⚙️ System Info")
st.sidebar.success("✅ Backend Connected")
st.sidebar.markdown("**FastAPI URL:** `http://127.0.0.1:8000`")
st.sidebar.markdown("**Embeddings Model:** Gemini (`models/embedding-001`)")
st.sidebar.markdown("**Vector DB:** FAISS")

st.sidebar.info("Developed by Daksh 🚀")
