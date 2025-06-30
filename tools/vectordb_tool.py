# ✅ FIXED TOOL DEFINITION

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from tools.vectordb_db import VectorDB

# Load the VectorDB from file
vector_db = VectorDB()
try:
    vector_db.load("scripts/vectordb_store")
except Exception as e:
    print(f"❌ Failed to load VectorDB: {e}")

# Define input schema properly
class QueryInput(BaseModel):
    query: str = Field(..., description="Query to find similar projects")

class VectorDBQueryTool(BaseTool):
    name: str = "VectorDB Query Tool"
    description: str = "Queries VectorDB for similar documents based on input query."
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        try:
            results = vector_db.query(query)
            return "\n".join(
                [f"Doc: {doc_id}, Sim: {dist:.4f}, Text: {text[:80]}..." for doc_id, text, dist in results]
            )
        except Exception as e:
            return f"❌ Tool failed with error: {e}"
