from crewai import Agent
from config.settings import gemini_llm
from tools.text_processor import TextProcessorTool
from tools.vectordb_tool import VectorDBQueryTool



vector_tool = VectorDBQueryTool()

project_context_agent = Agent(
    role="Context Extractor",
    goal="Understand project intent and related examples",
    backstory="You are an expert at identifying similar use cases.",
    tools=[vector_tool],  # ✅ Tool instance, not dict or call
    llm=gemini_llm,
    allow_delegation=False
)

