# ru-toxicity-detector

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![MIT License](https://img.shields.io/badge/License-MIT-green) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=en)

A simple toxicity detector.

## 🔍 About

### How It works

The model is built on [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) and trained using knowledge distillation from the more powerful [russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier). The training corpus used is the [Russian Language Toxic Comments](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments) dataset. The architecture features a hybrid approach: neural network embeddings are supplemented by signals from a built-in profanity dictionary (including an exceptions system). This allows the model to achieve high accuracy while maintaining a minimal size.

### Evaluation Results

The model was tested on an independent test set that was not used during training. To minimize false positives, the threshold was optimized for **Precision 95%+**.

| Metric                | Value    | Comment                                       |
|-----------------------|----------|-----------------------------------------------|
| **Accuracy**          | 0.99     | High value due to significant class imbalance |
| **Precision (Toxic)** | **0.95** | **95%** accuracy in classifying toxic content |
| **Recall (Toxic)**    | 0.65     | The model detects ~2/3 of all toxic messages  |
| **F1-score (Toxic)**  | 0.77     | Harmonic mean of precision and recall         |

The high **Precision (0.95)** ensures that the model almost never produces false positives. The lower Recall (0.65) is a deliberate trade-off to ensure a comfortable user experience.

## 📚 Usage

```python
from toxicity_detector import ToxicityDetector

# Create detector (optionally specify device)
detector = ToxicityDetector(device="cpu")

texts = [
    'Ты берега попутал?',                        # {'is_toxic': False, 'score': 0.2875}
    'Это правый берег реки, не путай с левым.',  # {'is_toxic': False, 'score': 0.0165}
    "Ты дуралей."                                # {'is_toxic': True, 'score': 0.9969}
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
Licensed under the **[MIT](LICENSE.txt)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
