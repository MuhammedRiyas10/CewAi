import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.doc_ids = []

    def load(self, path="vectordb_store"):
        index_path = os.path.join(path, "faiss_index.idx")
        meta_path = os.path.join(path, "meta.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("VectorDB files not found. Run the dummydbdata.py script first.")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.doc_ids = data["doc_ids"]

    def query(self, query_text: str, k=3) -> list:
        query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        results = [(self.doc_ids[i], self.documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

# ----------------- Test Script -----------------
if __name__ == "__main__":
    vector_db = VectorDB()
    vector_db.load("vectordb_store")
    print("✅ VectorDB loaded!")

    # Example queries:
    test_queries = [
        "Help me stay fit and track workouts",
        "Plan my trip to Japan with activities",
        "I want an app to manage my elderly parents' medication",
        "How can I spend less money and save more?",
        "Find me recipes with potatoes and cheese"
    ]

    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        results = vector_db.query(q, k=3)
        for doc_id, text, dist in results:
            print(f"- 📄 {doc_id} | Distance: {dist:.4f}")
            print(f"  🔹 {text}")
