import re

SPACE_PATTERN = re.compile(r'\s+')

def replace_yo_with_e(text: str) -> str:
    return text.replace('ё', 'е').replace('Ё', 'Е')

def remove_redundant_whitespaces(text: str) -> str:
    return SPACE_PATTERN.sub(' ', text).strip()

def preprocess_text(text: str) -> str:
    text = replace_yo_with_e(text)
    text = remove_redundant_whitespaces(text)
    return text
