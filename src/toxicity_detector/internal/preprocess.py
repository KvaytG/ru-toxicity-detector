from typing import Optional
from .text_utils import remove_redundant_whitespaces, replace_yo_with_e

def preprocess_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = replace_yo_with_e(text)
    text = remove_redundant_whitespaces(text)
    return text
