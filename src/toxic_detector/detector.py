import pathlib
import torch
from .internal.counter import ToxicityCounter
from .internal.model import Model
from .internal.vectorizer import vectorize

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = CURRENT_DIR / "resources" / "toxicity-model.pt"

class ToxicityDetector:
    def __init__(self):
        self._model = Model()
        self._toxicity_counter = ToxicityCounter()
        state_dict = torch.load(str(MODEL_PATH), map_location=torch.device('cpu'))
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def predict(self, text: str) -> int:
        t_vec = torch.tensor(vectorize(text)).squeeze(0).unsqueeze(0)
        d_score = torch.tensor([[self._toxicity_counter.get_score(text) * 0.33]], dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(t_vec, d_score)
            probability = torch.sigmoid(logits).item()
        return round(probability, 4)
