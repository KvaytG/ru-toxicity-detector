import re

SPACE_PATTERN = re.compile(r'\s+')

def remove_redundant_whitespaces(text: str) -> str:
    return SPACE_PATTERN.sub(' ', text).strip()
