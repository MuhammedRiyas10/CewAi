import streamlit as st
import os
import sqlite3
import uuid
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import Type
from pydantic import BaseModel, Field

# ---------- VectorDB Setup for Query Similarity Analysis ----------
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

    def query(self, query_text: str, k=3) -> list:
        query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        results = [(self.doc_ids[i], self.documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

# Initialize VectorDB
vector_db = VectorDB()

# ---------- Tool Schemas ----------
class TextInput(BaseModel):
    input_text: str = Field(..., description="Text input for processing")  # Changed field name to match _run parameter

class QueryInput(BaseModel):
    query: str = Field(..., description="Query for VectorDB similarity search")

# ---------- Custom Tools ----------
class TextProcessorTool(BaseTool):
    name: str = "Text Processor Tool"
    description: str = "Processes input text by converting to uppercase and counting words."
    args_schema: Type[BaseModel] = TextInput

    def _run(self, input_text: str) -> dict:
        return {"processed_text": input_text.upper(), "word_count": len(input_text.split())}

class VectorDBQueryTool(BaseTool):
    name: str = "VectorDB Query Tool"
    description: str = "Queries VectorDB for similar documents based on input query."
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        results = vector_db.query(query, k=3)
        output = "VectorDB Query Results:\n"
        for doc_id, text, distance in enumerate(results):
            output += f"- Doc ID: {doc_id}, Similarity Distance: {distance:.4f}, Text: {text[:100]}...\n"
        return output

# Tool Instances
text_processor_tool = TextProcessorTool()
vector_db_tool = VectorDBQueryTool()

# ---------- LLM Setup ----------
GEMINI_API_KEY = "AIzaSyCeV4_zg3m0Co6kyYRO5F9aFlwGL1ZRLic"  # Replace with your key

gemini_llm = LLM(model="gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)

# ---------- SQLite Database Setup (Unchanged) ----------
def init_db():
    conn = sqlite3.connect("feature_planning.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        metadata TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_journeys (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        journey_map TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS feature_plans (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        feature_plan TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_actions (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        action_list TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS refined_plans (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        feedback TEXT,
        refined_plan TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS vectordb_entries (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        text TEXT,
        created_at TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ---------- CrewAI Agents (Aligned with PDF) ----------
manager_agent = Agent(
    role="Manager",
    goal="Oversee feature planning and user action mapping, make decisions on task delegation, and facilitate inter-agent communication",
    backstory="Senior coordinator with expertise in managing multi-agent workflows for feature planning, ensuring alignment with user needs and project goals.",
    llm=gemini_llm,
    allow_delegation=True,
    verbose=True
)

project_context_agent = Agent(
    role="Project Context Interpreter",
    goal="Extract goals, audience, and outcomes from project metadata and user feedback",
    backstory="Expert in synthesizing project objectives and user needs from diverse inputs, leveraging vector embeddings for context.",
    llm=gemini_llm,
    tools=[text_processor_tool, vector_db_tool],
    verbose=True
)

user_journey_agent = Agent(
    role="User Journey Analyzer",
    goal="Analyze user journeys and map roles, intentions, and tasks using VectorDB for context-aware analysis",
    backstory="Specialist in identifying user behavior patterns and jobs-to-be-done, enhanced by vector-based retrieval.",
    llm=gemini_llm,
    tools=[text_processor_tool, vector_db_tool],
    verbose=True
)

task_plan_agent = Agent(
    role="Task Plan Generator",
    goal="Generate feature and task plans linked to user outcomes",
    backstory="Skilled in translating user journeys into developer-ready feature plans and tasks.",
    llm=gemini_llm,
    tools=[text_processor_tool],
    verbose=True
)

user_action_agent = Agent(
    role="User Action Mapper",
    goal="Generate contextual, inclusive user actions based on feature plans and journeys",
    backstory="Expert in crafting clear, user-focused actions with semantic understanding.",
    llm=gemini_llm,
    tools=[text_processor_tool],
    verbose=True
)

planner_refiner_agent = Agent(
    role="Planner Refiner",
    goal="Refine feature plans and user actions based on feedback",
    backstory="Proficient in iterative improvement of plans using peer or mentor feedback.",
    llm=gemini_llm,
    tools=[text_processor_tool],
    verbose=True
)

# ---------- Rest of the Code (Tasks, Streamlit UI, etc.) Remains Unchanged ----------
# [Include the create_tasks, strip_markdown, and Streamlit UI code from the previous version here]
# For brevity, assuming the rest of the code is unchanged and focuses on the task definitions and UI as provided earlier.

# ---------- CrewAI Tasks (Aligned with PDF) ----------
def create_tasks(project_metadata: str, user_feedback: str, project_id: str):
    manager_task = Task(
        description=f"""Oversee feature planning and user action mapping for project ID: {project_id}. 
        Inputs: project metadata ({project_metadata}), user feedback ({user_feedback}).
        - Query VectorDB for relevant past project data or feedback to inform task delegation.
        - Decide which agents should handle specific tasks based on input complexity and VectorDB results.
        - Facilitate communication (e.g., ensure Project Context Agent shares context with User Journey Agent).
        - Output a plain text summary of task assignments and decisions.""",
        agent=manager_agent,
        expected_output="Plain text summary of task assignments and decisions"
    )

    project_context_task = Task(
        description=f"""Extract goals, audience, and outcomes from project metadata: {project_metadata} and user feedback: {user_feedback}. 
        - Use VectorDB Query Tool to find similar project data for context.
        - Use Text Processor Tool to analyze input. 
        - Output a JSON-formatted project context with fields: objectives, audience, outcomes.""",
        agent=project_context_agent,
        expected_output="JSON-formatted project context with objectives, audience, and outcomes"
    )

    user_journey_task = Task(
        description=f"""Analyze user engagement based on project context and user feedback: {user_feedback}. 
        - Use VectorDB Query Tool to retrieve relevant user behavior patterns.
        - Use Text Processor Tool to cluster user roles and tasks. 
        - Output a JSON-formatted user journey map with user roles, intentions, and tasks.""",
        agent=user_journey_agent,
        expected_output="JSON-formatted user journey map with user roles, intentions, and tasks"
    )

    task_plan_task = Task(
        description="""Generate a feature and task plan based on user journey map and project context. 
        - Use Text Processor Tool to validate plan structure. 
        - Output a JSON-formatted feature plan with functions, tasks, and linked user outcomes.""",
        agent=task_plan_agent,
        expected_output="JSON-formatted feature plan with functions, tasks, and user outcomes"
    )

    user_action_task = Task(
        description="""Generate contextual, inclusive user actions based on feature plan and user journey map. 
        - Use Text Processor Tool for semantic understanding. 
        - Output a JSON-formatted list of at least 10 user actions per feature set.""",
        agent=user_action_agent,
        expected_output="JSON-formatted list of user actions"
    )

    planner_refiner_task = Task(
        description=f"""Refine feature plan and user actions based on feedback: {user_feedback}. 
        - Use Text Processor Tool to summarize feedback. 
        - Output a JSON-formatted refined feature plan and user action map, with a log of changes.""",
        agent=planner_refiner_agent,
        expected_output="JSON-formatted refined feature plan and user action map with change log"
    )

    return [manager_task, project_context_task, user_journey_task, task_plan_task, user_action_task, planner_refiner_task]

# ---------- Strip Markdown Function ----------
def strip_markdown(text: str) -> str:
    return re.sub(r'(?:json)?\s*([\s\S]*?)\s*', r'\1', text).strip()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Agentic AI Feature Planner", layout="wide")
st.title("🧠 Agentic AI Feature Planning and User Action Mapping")

# Custom CSS
st.markdown("""
    <style>
    .section-heading {
        font-size: 1.5em;
        font-weight: bold;
        color: #2E2E2E;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .text-box {
        border: 1px solid #CCCCCC;
        border-radius: 5px;
        padding: 10px;
        background-color: #F9F9F9;
        min-height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

with st.form("feature_form"):
    project_metadata = st.text_area("Project Metadata (e.g., goals, scope)", height=150)
    user_feedback = st.text_area("User Feedback/Surveys", height=150)
    submitted = st.form_submit_button("Generate Feature Plan")

# ---------- Example Inputs for Testing ----------
if not project_metadata and not user_feedback:
    project_metadata = """
    Project: Task Management App
    Goal: Create an app to help professionals manage tasks efficiently.
    Scope: Web-based platform with task creation, prioritization, and collaboration features.
    Target Audience: Project managers, freelancers, small business owners.
    """
    user_feedback = """
    - Users need a simple way to prioritize tasks daily.
    - Frustration with complex interfaces in existing tools.
    - Want collaboration features to assign tasks to team members.
    - Prefer instant notifications for task updates.
    """
    st.info("Using example inputs for a Task Management App.")

# ---------- Main Workflow ----------
if submitted:
    if not project_metadata or not user_feedback:
        st.error("Please fill in all fields.")
    else:
        # Generate unique project ID
        project_id = str(uuid.uuid4())

        # Save project metadata and feedback to SQLite and VectorDB
        conn = sqlite3.connect("feature_planning.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO projects (id, metadata, created_at) VALUES (?, ?, ?)",
                      (project_id, project_metadata, datetime.now()))
            c.execute("INSERT INTO vectordb_entries (id, project_id, text, created_at) VALUES (?, ?, ?, ?)",
                      (str(uuid.uuid4()), project_id, project_metadata + " " + user_feedback, datetime.now()))
            conn.commit()
        except Exception as e:
            st.error(f"Error saving to database: {e}")
        finally:
            conn.close()

        # Add project metadata and feedback to VectorDB
        vector_db.add_document(project_id, project_metadata + " " + user_feedback)

        # Run CrewAI workflow
        with st.spinner("Generating feature plan..."):
            tasks = create_tasks(project_metadata, user_feedback, project_id)
            crew = Crew(
                agents=[ project_context_agent, user_journey_agent, task_plan_agent, user_action_agent, planner_refiner_agent],
                tasks=tasks,
                process=Process.hierarchical,  # Hierarchical process with Manager Agent overseeing
                manager_agent=manager_agent,   # Manager Agent for decision-making
                verbose=True
            )
            try:
                result = crew.kickoff()
                st.write("CrewAI execution completed successfully.")
            except Exception as e:
                st.error(f"Error during CrewAI execution: {e}")
                st.stop()

            # Process task outputs
            outputs = {}
            for i, task_name in enumerate(["manager", "project_context", "user_journey", "task_plan", "user_action", "planner_refiner"]):
                try:
                    output = tasks[i].output
                    if hasattr(output, "raw") and isinstance(output.raw, str):
                        outputs[task_name] = strip_markdown(output.raw)
                    else:
                        st.warning(f"Task {task_name} output is not a valid string: {output}")
                        outputs[task_name] = f"Error: Invalid output format for task {task_name}"
                except Exception as e:
                    st.warning(f"Error processing output for task {task_name}: {e}")
                    outputs[task_name] = f"Error: Output processing failed: {e}"

            # Save results to database
            conn = sqlite3.connect("feature_planning.db")
            c = conn.cursor()
            try:
                if "project_context" in outputs and not outputs["project_context"].startswith("Error"):
                    c.execute("INSERT INTO projects (id, metadata, created_at) VALUES (?, ?, ?)",
                              (str(uuid.uuid4()), project_id, outputs["project_context"], datetime.now()))
                if "user_journey" in outputs and not outputs["user_journey"].startswith("Error"):
                    c.execute("INSERT INTO user_journeys (id, project_id, journey_map, created_at) VALUES (?, ?, ?, ?)",
                              (str(uuid.uuid4()), project_id, outputs["user_journey"], datetime.now()))
                if "task_plan" in outputs and not outputs["task_plan"].startswith("Error"):
                    c.execute("INSERT INTO feature_plans (id, project_id, feature_plan, created_at) VALUES (?, ?, ?, ?)",
                              (str(uuid.uuid4()), project_id, outputs["task_plan"], datetime.now()))
                if "user_action" in outputs and not outputs["user_action"].startswith("Error"):
                    c.execute("INSERT INTO user_actions (id, project_id, action_list, created_at) VALUES (?, ?, ?, ?)",
                              (str(uuid.uuid4()), project_id, outputs["user_action"], datetime.now()))
                if "planner_refiner" in outputs and not outputs["planner_refiner"].startswith("Error"):
                    c.execute("INSERT INTO refined_plans (id, project_id, feedback, refined_plan, created_at) VALUES (?, ?, ?, ?, ?)",
                              (str(uuid.uuid4()), project_id, user_feedback, outputs["planner_refiner"], datetime.now()))
                conn.commit()
            except Exception as e:
                st.error(f"Error saving to database: {e}")
            finally:
                conn.close()

            # Display results
            st.markdown('<div class="section-heading">Manager Decisions</div>', unsafe_allow_html=True)
            if "manager" in outputs and not outputs["manager"].startswith("Error"):
                st.text_area("", value=outputs["manager"], height=150, disabled=True, key="manager_output")
            else:
                st.warning("No manager output available.")

            st.markdown('<div class="section-heading">Project Context</div>', unsafe_allow_html=True)
            if "project_context" in outputs and not outputs["project_context"].startswith("Error"):
                st.text_area("", value=outputs["project_context"], height=150, disabled=True, key="project_context")
            else:
                st.warning("No project context output available.")

            st.markdown('<div class="section-heading">User Journey Map</div>', unsafe_allow_html=True)
            if "user_journey" in outputs and not outputs["user_journey"].startswith("Error"):
                st.text_area("", value=outputs["user_journey"], height=150, disabled=True, key="user_journey")
            else:
                st.warning("No user journey map output available.")

            st.markdown('<div class="section-heading">Feature and Task Plan</div>', unsafe_allow_html=True)
            if "task_plan" in outputs and not outputs["task_plan"].startswith("Error"):
                st.text_area("", value=outputs["task_plan"], height=200, disabled=True, key="task_plan")
            else:
                st.warning("No feature plan output available.")

            st.markdown('<div class="section-heading">User Actions</div>', unsafe_allow_html=True)
            if "user_action" in outputs and not outputs["user_action"].startswith("Error"):
                st.text_area("", value=outputs["user_action"], height=200, disabled=True, key="user_action")
            else:
                st.warning("No user actions output available.")

            st.markdown('<div class="section-heading">Refined Plan</div>', unsafe_allow_html=True)
            if "planner_refiner" in outputs and not outputs["planner_refiner"].startswith("Error"):
                feedback_parts = outputs["planner_refiner"].split("Refined Plan:", 1)
                feedback_summary = feedback_parts[0].strip()
                refined_plan = feedback_parts[1].strip() if len(feedback_parts) > 1 else "No refined plan provided."
                formatted_output = f"Feedback Summary:\n{feedback_summary}\n\nRefined Plan:\n{refined_plan}"
                st.text_area("", value=formatted_output, height=300, disabled=True, key="refined_plan")
            else:
                st.warning("No refined plan output available.")

# ---------- Example Output for Task Management App ----------
if submitted:
    st.subheader("Example Output for Task Management App")
    st.write("""
    **Manager Decisions**:
    - Queried VectorDB for similar task management projects; found 3 relevant entries.
    - Assigned Project Context Agent to extract goals and audience due to structured metadata.
    - Directed User Journey Agent to leverage VectorDB for user behavior patterns.
    - Ensured Task Plan Agent receives journey map for feature alignment.
    - Facilitated feedback loop to Planner Refiner Agent for iterative refinement.

    **Project Context**:
    ```json
    {
      "objectives": "Build a task management app for efficient task prioritization and collaboration",
      "audience": ["Project managers", "Freelancers", "Small business owners"],
      "outcomes": ["Simplified task prioritization", "Improved team collaboration", "Real-time updates"]
    }""")