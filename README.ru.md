# toxicity-detector

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![MIT License](https://img.shields.io/badge/License-MIT-green) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=ru)

Простой детектор токсичности.

## 📚 Использование

```python
from toxic_detector import ToxicityDetector

texts = [
    'Ты чего берега попутал?',                  # 0.9977
    'Это правый берег реки, не путай с левым.'  # 0.0141
]

detector = ToxicityDetector()
for idx, text in enumerate(texts, start=1):
    print(f'{idx}) {detector.predict(text)}')
```

## 📥 Установка

```bash
pip install git+https://github.com/KvaytG/toxicity-detector.git
```

## 📝 Лицензия

Распространяется по лицензии **[MIT](LICENSE.txt)**.

Проект использует компоненты с открытым исходным кодом. Сведения о лицензиях см. в **[pyproject.toml](pyproject.toml)** и на официальных ресурсах зависимостей.
