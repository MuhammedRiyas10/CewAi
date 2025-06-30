from crewai import Agent
from config.settings import gemini_llm
from tools.text_processor import TextProcessorTool

user_action_agent = Agent(
    role="User Action Mapper",
    goal="Generate contextual, inclusive user actions based on feature plans and journeys",
    backstory="Expert in crafting clear, user-focused actions with semantic understanding.",
    llm=gemini_llm,
    tools=[TextProcessorTool()],
    verbose=True
)
