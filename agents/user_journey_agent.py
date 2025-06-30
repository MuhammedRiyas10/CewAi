from crewai import Agent
from config.settings import gemini_llm
from tools.text_processor import TextProcessorTool
from tools.vectordb_tool import VectorDBQueryTool

user_journey_agent = Agent(
    role="User Journey Analyzer",
    goal="Analyze user journeys and map roles, intentions, and tasks using VectorDB for context-aware analysis",
    backstory="Specialist in identifying user behavior patterns and jobs-to-be-done, enhanced by vector-based retrieval.",
    llm=gemini_llm,
    tools=[TextProcessorTool(), VectorDBQueryTool()],
    verbose=True
)
