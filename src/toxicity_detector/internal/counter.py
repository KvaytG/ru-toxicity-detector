import math
import re
import json
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
VOCAB_PATH = CURRENT_DIR.parent / "resources" / "vocab.json"

WORDS_PATTERN = re.compile(r'\b\w+\b')

class ToxicityCounter:
    def __init__(self):
        with open(str(VOCAB_PATH), 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.bad_regex = re.compile('|'.join(data.get('profanity')), re.IGNORECASE)
        self.clean_regex = re.compile('|'.join(data.get('exceptions')), re.IGNORECASE)

    def get_score(self, text: str) -> float:
        if not text:
            return 0.0
        words = WORDS_PATTERN.findall(text.lower())
        bad_count = 0
        for word in words:
            if self.bad_regex.fullmatch(word) and (not self.clean_regex.fullmatch(word)):
                bad_count += 1
        return 1.0 - math.pow(0.8, bad_count)
