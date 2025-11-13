# ---------------------------------------------------
# gemini_embeddings.py
# Custom Gemini Embedding Wrapper for LangChain + FAISS
# ---------------------------------------------------

import google.generativeai as genai
import os
from dotenv import load_dotenv


# --------------------------------------------
# ⚙️ Load Gemini API key from .env file
# --------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

genai.configure(api_key=api_key)


# --------------------------------------------
# 🧠 GeminiEmbeddings Class
# --------------------------------------------
class GeminiEmbeddings:
    """
    Custom embedding class for Google Gemini API.
    Compatible with LangChain vector stores (FAISS, Chroma, etc.)
    """

    def __init__(self, model_name: str = "models/embedding-001"):
        """
        Initialize the Gemini embedding model.
        Args:
            model_name: The embedding model to use (default: "models/embedding-001")
        """
        self.model = model_name

    def embed_documents(self, texts):
        """
        Generate embeddings for multiple document chunks.
        Args:
            texts (List[str]): List of text chunks.
        Returns:
            List[List[float]]: List of vector embeddings.
        """
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(model=self.model, content=text)
                embeddings.append(result["embedding"])
            except Exception as e:
                print(f"⚠️ Embedding failed for one document: {e}")
        return embeddings

    def embed_query(self, query: str):
        """
        Generate embedding for a single query string.
        Args:
            query (str): User query.
        Returns:
            List[float]: Vector embedding for the query.
        """
        try:
            result = genai.embed_content(model=self.model, content=query)
            return result["embedding"]
        except Exception as e:
            print(f"⚠️ Query embedding failed: {e}")
            return []

    def __call__(self, text: str):
        """
        Makes the object callable (so FAISS can call it directly).
        Args:
            text (str): Input text or query.
        Returns:
            List[float]: Embedding vector.
        """
        return self.embed_query(text)
