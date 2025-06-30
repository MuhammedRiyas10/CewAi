import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.doc_ids = []

    def add_document(self, doc_id: str, text: str):
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        self.index.add(np.array([embedding]))
        self.documents.append(text)
        self.doc_ids.append(doc_id)

    def save(self, path="vectordb_store"):
        if not os.path.exists(path):
            os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, "faiss_index.idx"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "documents": self.documents
            }, f)

# ----------------- Dummy Project Data -----------------
dummy_projects = [
    ("proj-finance-001", "A mobile budgeting app that tracks spending habits and suggests savings goals."),
    ("proj-fitness-002", "Fitness tracker with smart AI workout plans, sleep monitoring, and hydration alerts."),
    ("proj-edu-003", "An AI-powered learning platform that generates study plans and explains homework in simple terms."),
    ("proj-travel-004", "A trip planning assistant that builds personalized itineraries based on mood and location."),
    ("proj-task-005", "A team task manager app with drag-and-drop task boards, auto-reminders, and progress reports."),
    ("proj-health-006", "Habit tracker to log water, nutrition, and sleep. Integrates with wearables for insights."),
    ("proj-lang-007", "Language learning app using voice AI to correct pronunciation and engage in live conversations."),
    ("proj-mental-008", "Mental wellness journaling app with mood tracking and AI-generated affirmations."),
    ("proj-recruit-009", "An AI recruiter tool that parses resumes, filters candidates, and auto-schedules interviews."),
    ("proj-ecom-010", "Smart shopping assistant that compares prices, recommends deals, and tracks deliveries."),
    ("proj-news-011", "News aggregator app using AI to summarize articles and explain complex topics."),
    ("proj-food-012", "Recipe suggestion app based on available ingredients, calorie limits, and taste preferences."),
    ("proj-pet-013", "Pet care manager that logs feeding schedules, vet appointments, and health alerts."),
    ("proj-senior-014", "Assisted living support app for elderly users, with reminders, emergency calls, and daily tips."),
    ("proj-climate-015", "Sustainability planner to track your carbon footprint and offer weekly eco-friendly goals."),
]

# ----------------- Add to VectorDB and Save -----------------
vector_db = VectorDB()

for doc_id, text in dummy_projects:
    vector_db.add_document(doc_id, text)

vector_db.save("vectordb_store")
print("✅ VectorDB populated with 15 dummy projects and saved to 'vectordb_store'")

