# AI Document Semantic Search (Gemini + FAISS)

Self-hosted semantic search for PDFs and TXT files. Streamlit frontend + FastAPI backend with FAISS for vector search. Embeddings can use Google Gemini (wrapper included) or be swapped for another provider.

## Features
- Upload and index PDF / TXT files
- Chunking and vector indexing with FAISS
- Query interface via Streamlit that returns top matching snippets
- FastAPI endpoints for ingestion and search
- Gemini embeddings wrapper included (configurable)

## Tech stack
- Frontend: Streamlit
- Backend: FastAPI + Uvicorn
- Vector DB: FAISS (faiss-cpu)
- Embeddings: Gemini (or alternative)
- PDF parsing: pdfminer.six, PyPDF2
- Environment: Windows PowerShell instructions included

## Quick setup (Windows PowerShell)
1. Copy and configure env:
   ```
   copy .env.example .env
   # Edit .env and add GEMINI_API_KEY and other secrets
   ```
2. Create virtual env and install:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Start backend (from repo root):
   ```
   uvicorn main:app --reload
   ```
4. Run frontend:
   ```
   streamlit run app.py
   ```

## Usage
- Upload PDFs/TXT via Streamlit UI.
- Enter a query and select number of results.
- Results show source file and matching text snippets.

## Notes
- Do NOT commit `.env` or the FAISS index files. Use `.env.example` as a template.
- Large index files should be stored outside the repository (cloud storage or Git LFS).
- If you change embedding provider, update backend to use the corresponding wrapper.

## Contributing
- Ensure secrets are excluded.
- Add tests and update requirements when adding new dependencies.

## License
Choose an appropriate license and add LICENSE file before public release.