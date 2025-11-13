from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_FOLDER = "faiss_index"
model_name = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=model_name)

# Load FAISS index
db = FAISS.load_local(INDEX_FOLDER, embedder, allow_dangerous_deserialization=True)

query = input("\n🔍 Enter your query: ")
results = db.similarity_search(query, k=5)

if not results:
    print("⚠️ No results found. Try a different query.")
else:
    print(f"\n🎯 Top results for: '{query}'\n")
    for i, r in enumerate(results, start=1):
        print(f"Rank {i}")
        print(f"📘 Source: {r.metadata['source']}")
        print(f"📝 Text: {r.page_content[:300]}...\n{'-'*60}")
