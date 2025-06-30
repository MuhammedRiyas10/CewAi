from crewai import Agent
from config.settings import gemini_llm
from tools.text_processor import TextProcessorTool

planner_refiner_agent = Agent(
    role="Planner Refiner",
    goal="Refine feature plans and user actions based on feedback",
    backstory="Proficient in iterative improvement of plans using peer or mentor feedback.",
    llm=gemini_llm,
    tools=[TextProcessorTool()],
    verbose=True
)
