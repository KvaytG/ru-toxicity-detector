
# ru-toxicity-detector

![US](https://kvaytg.ru/common/flags/us-21x16.svg) **English** | [![RU](https://kvaytg.ru/common/flags/ru-21x16.svg) Русский](README.ru.md)

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue) ![PolyForm License](https://img.shields.io/badge/License-PolyForm-blue) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=en)

A simple toxicity detector.

## 🔍 About

### How It works

The model is built on [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) and trained using knowledge distillation from the more powerful [russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier). The architecture features a hybrid approach: neural network embeddings are supplemented by signals from a built-in profanity dictionary (including an exceptions system). This allows the model to achieve high accuracy while maintaining a minimal size.

### Evaluation Results

The model was tested on an independent test set of **102,308** lines that was completely unseen during training. To minimize false positives, the classification threshold was dynamically optimized for **Precision 95%** on the validation set and saved directly inside the model weights file.

| Metric                | Value    | Comment                                       |
|-----------------------|----------|-----------------------------------------------|
| **Accuracy**          | 1.00     | High value due to significant class imbalance |
| **Precision (Toxic)** | **0.95** | **95%** accuracy in classifying toxic content |
| **Recall (Toxic)**    | 0.70     | The model detects ~70% of all toxic messages  |
| **F1-score (Toxic)**  | 0.80     | Harmonic mean of precision and recall         |

The high **Precision (0.95)** ensures that the model almost never produces false positives. The lower Recall (0.70) is a deliberate trade-off to ensure a comfortable user experience without aggressive over-blocking.

## 📚 Usage
```python
from toxicity_detector import ToxicityDetector

# Create detector (optionally specify device)
detector = ToxicityDetector(device="cpu")

texts = [
    "Я люблю тебя",  # {'is_toxic': False, 'score': 0.0122}
    "Ты дуралей"     # {'is_toxic': True,  'score': 0.822}
]

# Predict one by one
for idx, text in enumerate(texts, start=1):
    print(f"{idx}) {detector.predict(text)}")

# Predict batch (faster for multiple texts)
results = detector.predict_batch(texts)
for idx, res in enumerate(results, start=1):
    print(f"{idx}) {res}")
```

## 📥 Installation
```bash
pip install git+https://github.com/KvaytG/ru-toxicity-detector.git
```

## 📝 License
Licensed under the **[PolyForm Noncommercial](LICENSE.md)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
