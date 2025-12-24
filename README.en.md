# toxicity-detector

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![MIT License](https://img.shields.io/badge/License-MIT-green) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=en)

A simple toxicity detector.

## 📚 Usage

```python
from toxicity_detector import ToxicityDetector

texts = [
    'Ты чего берега попутал?',                  # 0.9977
    'Это правый берег реки, не путай с левым.'  # 0.0141
]

detector = ToxicityDetector()
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
