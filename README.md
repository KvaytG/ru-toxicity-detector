# ru-toxicity-detector

**English** | [Русский](README.ru.md)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-PolyForm-blue) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=en)

Russian text toxicity detection powered by [KvaytG/rubert-tiny2-toxic-detector](https://huggingface.co/KvaytG/rubert-tiny2-toxic-detector).

## How it works

The package runs the pretrained sequence-classification model directly through Hugging Face Transformers.

The default toxicity threshold is `0.70`. A text is marked toxic when its predicted toxicity probability is strictly greater than the configured threshold. The returned `score` is the probability rounded to four decimal places.

The model and tokenizer are downloaded from Hugging Face the first time the detector is initialized and cached for later use. An internet connection is needed on first load unless `model_name` points to a local copy.

## Installation

```bash
pip install git+https://github.com/KvaytG/ru-toxicity-detector.git
```

For local development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from toxicity_detector import ToxicityDetector

# Create detector (optionally specify device)
detector = ToxicityDetector(device="cpu")

texts = [
    "Я люблю тебя",  # {'is_toxic': False, 'score': 0.0119}
    "Пошёл нахуй"    # {'is_toxic': True,  'score': 0.9736}
]

# Predict one by one
for idx, text in enumerate(texts, start=1):
    print(f"{idx}) {detector.predict(text)}")

# Predict batch (faster for multiple texts)
results = detector.predict_batch(texts)
for idx, res in enumerate(results, start=1):
    print(f"{idx}) {res}")
```

## Threshold and device

```python
# Use a stricter threshold to reduce false positives or a lower one to
# increase sensitivity. The appropriate value depends on your use case.
detector = ToxicityDetector(threshold=0.80, device="cpu")

# If a compatible CUDA-enabled PyTorch installation is available:
detector = ToxicityDetector(device="cuda")
```

Constructor options:

| Argument     | Default                                | Description                                      |
|--------------|----------------------------------------|--------------------------------------------------|
| `threshold`  | `0.70`                                 | Toxicity probability cutoff, from `0.0` to `1.0` |
| `device`     | `"cpu"`                                | PyTorch inference device                         |
| `model_name` | `"KvaytG/rubert-tiny2-toxic-detector"` | Hugging Face model ID or local model directory   |
| `max_length` | `128`                                  | Maximum token sequence length                    |
| `batch_size` | `64`                                   | Number of texts per inference batch              |

Both `predict(text)` and `predict_batch(texts)` return dictionaries with the same fields:

- `is_toxic`: whether the probability is strictly greater than the configured threshold.
- `score`: predicted toxicity probability, rounded to four decimal places.

## Model and evaluation

See the [model card](https://huggingface.co/KvaytG/rubert-tiny2-toxic-detector) for training details, evaluation metrics, the recommended threshold, and the model's Apache-2.0 license.

## Project license

This project's source code is distributed under the [PolyForm Noncommercial License](LICENSE.md). The pretrained model and its usage are subject to the model's own license and terms.
