# toxicity-detector

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![MIT License](https://img.shields.io/badge/License-MIT-green) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=en)

A simple toxicity detector.

## 🔍 About

### How It Works

The model is built on [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) and trained using knowledge distillation from the more powerful [russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier). The training corpus used is the [Russian Language Toxic Comments](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments) dataset. The architecture features a hybrid approach: neural network embeddings are supplemented by signals from a built-in profanity dictionary (including an exceptions system). This allows the model to achieve high accuracy while maintaining a minimal size.

### Quality Metrics

The model was tested on an independent test set that was not used during training. To minimize false positives, the threshold was optimized for **Precision 95%+**.

| Metric                  | Value    |
|-------------------------|----------|
| **Accuracy**            | **0.89** |
| **Precision (Toxic)**   | **0.98** |
| **Recall (Toxic)**      | **0.67** |
| **F1-score (Weighted)** | **0.89** |

The high **Precision (0.98)** ensures that the model almost never produces false positives. The lower Recall (0.67) is a deliberate trade-off to ensure a comfortable user experience.

## 📚 Usage

```python
from toxicity_detector import ToxicityDetector

texts = [
    'Ты чего, берега попутал?',                  # {'is_toxic': True, 'confidence': 0.5646}
    'Это правый берег реки, не путай с левым.',  # {'is_toxic': False, 'confidence': 0.0341}
    "Ты дурачьё."                                # {'is_toxic': True, 'confidence': 0.9328}
]

detector = ToxicityDetector(0.5)
for idx, text in enumerate(texts, start=1):
    print(f'{idx}) {detector.predict(text)}')
```

## 📥 Installation
```bash
pip install git+https://github.com/KvaytG/toxicity-detector.git
```

## 📝 License
Licensed under the **[MIT](LICENSE.txt)** license.

This project uses open-source components. For license details see **[pyproject.toml](pyproject.toml)** and dependencies' official websites.
