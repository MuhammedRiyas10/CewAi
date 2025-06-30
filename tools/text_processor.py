from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class TextInput(BaseModel):
    input_text: str = Field(...)

class TextProcessorTool(BaseTool):
    name: str = "Text Processor Tool"
    description: str = "Processes input text by converting to uppercase and counting words."
    args_schema: Type[BaseModel] = TextInput
    ...

    def _run(self, input_text: str):
        return {"processed_text": input_text.upper(), "word_count": len(input_text.split())}
