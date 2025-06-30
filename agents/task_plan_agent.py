from crewai import Agent
from config.settings import gemini_llm
from tools.text_processor import TextProcessorTool

task_plan_agent = Agent(
    role="Task Plan Generator",
    goal="Generate feature and task plans linked to user outcomes",
    backstory="Skilled in translating user journeys into developer-ready feature plans and tasks.",
    llm=gemini_llm,
    tools=[TextProcessorTool()],
    verbose=True
)
