from crewai import LLM

GEMINI_API_KEY = "AIzaSyBgyh7agEgAJkDXND2u0V1MQK69ELjBs4w"

gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=GEMINI_API_KEY
)
