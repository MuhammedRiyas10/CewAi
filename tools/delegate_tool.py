# tools/delegate_tool.py

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class DelegateInput(BaseModel):
    task: str = Field(..., description="The task to delegate")
    context: str = Field(..., description="The context to perform the task")
    coworker: str = Field(..., description="The coworker to delegate the task to")

class DelegateTool(BaseTool):
    name = "Delegate work to coworker"
    description = "Delegate a task to a coworker with context"
    args_schema: Type[BaseModel] = DelegateInput

    def _run(self, task: str, context: str, coworker: str) -> str:
        return f"✅ Task '{task}' delegated to {coworker}.\nContext: {context}"
