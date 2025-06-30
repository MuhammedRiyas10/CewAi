# tasks/create_tasks.py

from crewai import Task
from tools.vectordb_tool import VectorDBQueryTool
from tools.text_processor import TextProcessorTool
from agents import (
    manager_agent,
    project_context_agent,
    user_journey_agent,
    task_plan_agent,
    user_action_agent,
    planner_refiner_agent
)

# ✅ Instantiate tools
vector_tool = VectorDBQueryTool()
text_tool = TextProcessorTool()

def create_tasks(project_metadata: str, user_feedback: str, project_id: str):
    # ✅ Manager Task (No tools required here unless you explicitly want them)
    manager_task = Task(
        description=f"""
You are the manager for a feature planning system. You will:

- Oversee the planning process for project ID: {project_id}
- Decide which agents should do what
- Understand this project: {project_metadata}
- Use feedback: {user_feedback}
- Query VectorDB for similar context (if needed)
- Output a plain text summary of decisions and assignments.
""",
        agent=manager_agent,
        expected_output="Plain text summary of task assignments and decisions"
    )

    # ✅ Project Context Extraction
    project_context_task = Task(
        description=f"""
Use the project metadata below to query VectorDB and retrieve top 3 similar past project documents:
Metadata:
{project_metadata}

Then extract inspiration for current project context.
""",
        agent=project_context_agent,
        tools=[vector_tool],
        expected_output="Top 3 relevant projects for inspiration and design direction."
    )

    # ✅ User Journey Mapping
    user_journey_task = Task(
        description=f"""
Analyze user engagement patterns based on:
- Project context
- User feedback: {user_feedback}

Use tools to cluster behaviors and generate a user journey map.
""",
        agent=user_journey_agent,
        tools=[vector_tool, text_tool],
        expected_output="JSON-formatted user journey map with user roles, intentions, and tasks"
    )

    # ✅ Task & Feature Planning
    task_plan_task = Task(
        description="""
Create a feature and task plan based on:
- User journey map
- Project context

Use text analysis to validate plan.
""",
        agent=task_plan_agent,
        tools=[text_tool],
        expected_output="JSON-formatted feature plan with functions, tasks, and user outcomes"
    )

    # ✅ User Action Mapping
    user_action_task = Task(
        description="""
Generate actionable user actions for each feature in the plan.
Use semantics for accuracy.
""",
        agent=user_action_agent,
        tools=[text_tool],
        expected_output="JSON-formatted list of user actions"
    )

    # ✅ Refiner Task
    planner_refiner_task = Task(
        description=f"""
Take the original plan and user actions and refine them based on:
- User feedback: {user_feedback}

Include change logs and summary.
""",
        agent=planner_refiner_agent,
        tools=[text_tool],
        expected_output="JSON-formatted refined feature plan and user action map with change log"
    )

    return [
        manager_task,
        project_context_task,
        user_journey_task,
        task_plan_task,
        user_action_task,
        planner_refiner_task
    ]
