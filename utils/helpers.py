import re

def strip_markdown(text: str) -> str:
    return re.sub(r'(?:json)?\s*([\s\S]*?)\s*', r'\1', text).strip()
