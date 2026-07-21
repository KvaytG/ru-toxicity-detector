import pathlib
import torch
from .internal.constants import THRESHOLD
from .internal.counter import ToxicityCounter
from .internal.model import Model
from .internal.vectorizer import vectorize, vectorize_batch

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = CURRENT_DIR / "resources" / "toxicity-model.pt"


class ToxicityDetector:
    def __init__(self, threshold: float | None = None, device: str = "cpu"):
        self._device = torch.device(device)
        self._model = Model().to(self._device)
        self._toxicity_counter = ToxicityCounter()
        state_dict = torch.load(str(MODEL_PATH), map_location=self._device)
        model_threshold = None
        if "threshold" in state_dict:
            model_threshold = float(state_dict.pop("threshold").item())
        if threshold is not None:
            self.threshold = threshold
        elif model_threshold is not None:
            self.threshold = model_threshold
        else:
            self.threshold = THRESHOLD
        self._model.load_state_dict(state_dict)
        self._model.eval()

    @staticmethod
    def _empty_result():
        return {
            "is_toxic": False,
            "score": 0.0
        }

    def predict(self, text: str) -> dict:
        if not text:
            return self._empty_result()
        vec = vectorize(text).to(self._device)
        raw_score = self._toxicity_counter.get_score(text)
        d_score = torch.tensor([[raw_score]], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            logits = self._model(vec, d_score)
            prob = torch.sigmoid(logits).item()
        return {
            "is_toxic": prob >= self.threshold,
            "score": round(prob, 4)
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        mask_empty = [not t for t in texts]
        valid_texts = [t for t in texts if t]
        embeddings = vectorize_batch(valid_texts).to(self._device) if valid_texts else None
        dict_scores = [
            [self._toxicity_counter.get_score(t)] for t in texts if t
        ]
        results = []
        idx = 0
        if embeddings is not None:
            d_score = torch.tensor(dict_scores, dtype=torch.float32, device=self._device)
            with torch.no_grad():
                logits = self._model(embeddings, d_score)
                probs = torch.sigmoid(logits).squeeze(1).tolist()
        else:
            probs = []
        for is_empty in mask_empty:
            if is_empty:
                results.append(self._empty_result())
            else:
                p = probs[idx]
                results.append({
                    "is_toxic": p >= self.threshold,
                    "score": round(p, 4)
                })
                idx += 1
        return results
