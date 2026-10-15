import math
import sys
import types
import pytest
import torch
from toxicity_detector import ToxicityDetector


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        ids = {"negative": 0, "middle": 1, "positive": 2}
        values = [ids[text] for text in texts]
        return {
            "input_ids": torch.tensor([[value] for value in values], dtype=torch.long),
            "attention_mask": torch.ones((len(values), 1), dtype=torch.long),
        }


class FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self

    def __call__(self, input_ids, attention_mask):
        logits_by_id = torch.tensor(
            [-2.1972246, 0.0, 2.1972246], dtype=torch.float32, device=input_ids.device
        )
        return types.SimpleNamespace(logits=logits_by_id[input_ids.reshape(-1)].reshape(-1, 1))


def install_fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: FakeTokenizer())
    module.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda name: FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.fixture

def detector(monkeypatch):
    install_fake_transformers(monkeypatch)
    return ToxicityDetector()


def test_predict_returns_probability_and_threshold_result(detector):
    positive = detector.predict("positive")
    negative = detector.predict("negative")

    assert positive["is_toxic"] is True
    assert positive["score"] == pytest.approx(0.9, abs=1e-4)
    assert negative["is_toxic"] is False
    assert negative["score"] == pytest.approx(0.1, abs=1e-4)


def test_custom_threshold(monkeypatch):
    install_fake_transformers(monkeypatch)
    detector = ToxicityDetector(threshold=0.95)
    assert detector.predict("positive")["is_toxic"] is False


def test_batch_preserves_order_and_empty_text(detector):
    results = detector.predict_batch(["positive", "", "negative", "   ", "middle"])

    assert len(results) == 5
    assert results[0]["is_toxic"] is True
    assert results[1] == {"is_toxic": False, "score": 0.0}
    assert results[2]["is_toxic"] is False
    assert results[3] == {"is_toxic": False, "score": 0.0}
    assert results[4]["is_toxic"] is False


def test_batch_chunks_are_supported(monkeypatch):
    install_fake_transformers(monkeypatch)
    detector = ToxicityDetector(batch_size=1)
    results = detector.predict_batch(["positive", "negative", "positive"])
    assert [result["is_toxic"] for result in results] == [True, False, True]


def test_blank_text_does_not_need_model_inference(detector):
    assert detector.predict("  \n") == {"is_toxic": False, "score": 0.0}
    assert detector.predict_batch([]) == []
    assert detector.predict_batch(["", " "]) == [
        {"is_toxic": False, "score": 0.0},
        {"is_toxic": False, "score": 0.0},
    ]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.inf, math.nan])
def test_rejects_invalid_threshold(monkeypatch, threshold):
    install_fake_transformers(monkeypatch)
    with pytest.raises(ValueError):
        ToxicityDetector(threshold=threshold)


def test_rejects_non_string_input(detector):
    with pytest.raises(TypeError):
        detector.predict(42)
    with pytest.raises(TypeError):
        detector.predict_batch("not a list")
    with pytest.raises(TypeError):
        detector.predict_batch(["ok", None])
