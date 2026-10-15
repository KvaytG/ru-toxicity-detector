from __future__ import annotations
import math
import torch
from typing import Sequence
from ._internal import MODEL_NAME, MAX_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_THRESHOLD


class ToxicityDetector:
    """Detect toxic Russian text with ``KvaytG/rubert-tiny2-toxic-detector``.

    The pretrained model and tokenizer are downloaded from Hugging Face on the
    first initialization and cached by ``transformers`` for subsequent runs.

    Args:
        threshold: Toxicity probability cutoff in the inclusive range [0, 1].
        device: PyTorch device, for example ``"cpu"`` or ``"cuda"``.
        model_name: Hugging Face model ID or a local model directory.
        max_length: Maximum token sequence length used by the tokenizer.
        batch_size: Number of texts passed to the model in each inference batch.
    """
    def __init__(
        self,
        threshold: float | None = None,
        device: str = "cpu",
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be a finite number between 0.0 and 1.0")
        if max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._device = torch.device(device)
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ToxicityDetector requires transformers. Install project dependencies "
                "with `pip install -e .` or install `transformers` directly."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _empty_result() -> dict[str, bool | float]:
        return {"is_toxic": False, "score": 0.0}

    def _predict_non_empty(self, texts: Sequence[str]) -> list[dict[str, bool | float]]:
        """Predict a non-empty collection of non-blank texts."""
        results: list[dict[str, bool | float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [text.strip() for text in texts[start : start + self._batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                outputs = self._model(**encoded)
                probabilities = torch.sigmoid(outputs.logits.reshape(-1)).detach().cpu().tolist()
            results.extend(
                {
                    "is_toxic": float(probability) > self.threshold,
                    "score": round(float(probability), 4),
                }
                for probability in probabilities
            )
        return results

    def predict(self, text: str) -> dict[str, bool | float]:
        """Return the toxicity decision and probability for one text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return self._empty_result()
        return self._predict_non_empty([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[dict[str, bool | float]]:
        """Predict a sequence of texts, preserving input order and empty items."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings, not a single string")
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("every item in texts must be a string")
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return [self._empty_result() for _ in texts]
        predictions = iter(self._predict_non_empty(valid_texts))
        return [next(predictions) if text.strip() else self._empty_result() for text in texts]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(threshold={self.threshold}, "
            f"device={str(self._device)!r})"
        )
