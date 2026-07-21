
# ru-toxicity-detector

[![US](https://kvaytg.ru/common/flags/us-21x16.svg) English](README.md) | ![RU](https://kvaytg.ru/common/flags/ru-21x16.svg) **Русский**

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue) ![PolyForm License](https://img.shields.io/badge/License-PolyForm-blue) [![Sponsor](https://img.shields.io/badge/Поддержать-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=ru)

Простой детектор токсичности.

## 🔍 О проекте

### Как это работает

Модель построена на базе [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) и обучена методом дистилляции знаний из более мощного классификатора [russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier). Особенностью архитектуры является гибридный подход: нейросетевые эмбеддинги дополняются сигналом от встроенного словаря ненормативной лексики с системой исключений, что позволяет достичь высокой точности при минимальном размере модели.

### Результаты тестирования

Модель была протестирована на независимой тестовой выборке из **102 308** строк, которая не участвовала в обучении. Для минимизации ложных срабатываний классификационный порог был динамически оптимизирован под **Precision 95%** на валидационной выборке и сохранен прямо внутри файла весов модели.

| Метрика               | Значение | Комментарий                                        |
|-----------------------|----------|----------------------------------------------------|
| **Accuracy**          | 1.00     | Высокий показатель обусловлен дисбалансом классов  |
| **Precision (Toxic)** | **0.95** | **95%** точность классификации токсичного контента |
| **Recall (Toxic)**    | 0.70     | Модель находит ~70% всех токсичных сообщений       |
| **F1-score (Toxic)**  | 0.80     | Гармоническое среднее между точностью и полнотой   |

Высокий показатель **Precision (0.95)** гарантирует, что модель практически не допускает ложных срабатываний. Низкий Recall (0.70) является осознанным компромиссом для обеспечения комфортного пользовательского опыта без чрезмерной блокировки.

## 📚 Использование
```python
from toxicity_detector import ToxicityDetector

# Создать детектор (можно указывать устройство)
detector = ToxicityDetector(device="cpu")

texts = [
    "Я люблю тебя",  # {'is_toxic': False, 'score': 0.0122}
    "Ты дуралей"     # {'is_toxic': True,  'score': 0.822}
]

# Предсказания по одному тексту
for idx, text in enumerate(texts, start=1):
    print(f"{idx}) {detector.predict(text)}")

# Предсказания батчем (быстрее для нескольких текстов)
results = detector.predict_batch(texts)
for idx, res in enumerate(results, start=1):
    print(f"{idx}) {res}")
```

## 📥 Установка
```bash
pip install git+https://github.com/KvaytG/ru-toxicity-detector.git
```

## 📝 Лицензия
Распространяется по лицензии **[PolyForm Noncommercial](LICENSE.md)**.

Проект использует компоненты с открытым исходным кодом. Сведения о лицензиях см. в **[pyproject.toml](pyproject.toml)** и на официальных ресурсах зависимостей.
