from crewai import Agent
from config.settings import gemini_llm
from tools.delegate_tool import DelegateTool
delegate_tool = DelegateTool()

manager_agent = Agent(
    role="Manager",
    goal="Coordinate tasks among all agents",
    backstory="You manage and delegate tasks to the team.",
    tools=[delegate_tool],  # ✅ Pass tool instance, not a dict
    llm=gemini_llm,
    allow_delegation=True
)