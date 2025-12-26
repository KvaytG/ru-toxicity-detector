import pathlib
import torch
from .internal.counter import ToxicityCounter
from .internal.model import Model
from .internal.vectorizer import vectorize

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = CURRENT_DIR / "resources" / "toxicity-model.pt"


class ToxicityDetector:
    def __init__(self, threshold: float = 0.8555):
        self._model = Model()
        self._toxicity_counter = ToxicityCounter()
        self.threshold = threshold
        state_dict = torch.load(str(MODEL_PATH), map_location=torch.device('cpu'))
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def predict(self, text: str) -> dict:
        vec = vectorize(text)
        t_vec = torch.tensor(vec, dtype=torch.float32)
        if t_vec.dim() == 1:
            t_vec = t_vec.unsqueeze(0)
        raw_words_score = self._toxicity_counter.get_score(text)
        d_score = torch.tensor([[raw_words_score]], dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(t_vec, d_score)
            probability = torch.sigmoid(logits).item()
        is_toxic = probability >= self.threshold
        return {
            "is_toxic": is_toxic,
            "confidence": round(probability, 4)
        }
