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

    def load(self, path="scripts/vectordb_store"):
        index_path = os.path.join(path, "faiss_index.idx")
        meta_path = os.path.join(path, "meta.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("VectorDB files not found. Run dummydbdata.py to generate.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.doc_ids = data["doc_ids"]

    def query(self, query_text: str, k=3):
        query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [(self.doc_ids[i], self.documents[i], distances[0][j]) for j, i in enumerate(indices[0])]