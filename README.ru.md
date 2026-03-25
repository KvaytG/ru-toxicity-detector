# ru-toxicity-detector

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![MIT License](https://img.shields.io/badge/Лицензия-MIT-green) [![Sponsor](https://img.shields.io/badge/Поддержать-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=ru) [![Telegram](https://img.shields.io/badge/Telegram-Канал-blue?logo=telegram)](https://t.me/kvaytgk)

Простой детектор токсичности.

## 🔍 О проекте

### Как это работает

Модель построена на базе [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) и обучена методом дистилляции знаний из более мощного классификатора [russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier). В качестве обучающего корпуса использован датасет [Russian Language Toxic Comments](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments). Особенностью архитектуры является гибридный подход: нейросетевые эмбеддинги дополняются сигналом от встроенного словаря ненормативной лексики с системой исключений, что позволяет достичь высокой точности при минимальном размере модели.

### Результаты тестирования

Модель была протестирована на независимой тестовой выборке, которая не участвовала в обучении. Для минимизации ложных срабатываний порог был оптимизирован под **Precision 95%+**.

| Метрика               | Значение | Комментарий                                        |
|-----------------------|----------|----------------------------------------------------|
| **Accuracy**          | 0.99     | Высокий показатель обусловлен дисбалансом классов  |
| **Precision (Toxic)** | **0.95** | **95%** точность классификации токсичного контента |
| **Recall (Toxic)**    | 0.65     | Модель детектирует ~2/3 всех токсичных сообщений   |
| **F1-score (Toxic)**  | 0.77     | Гармоническое среднее между точностью и полнотой   |

Высокий показатель **Precision (0.95)** гарантирует, что модель практически не допускает ложных срабатываний. Низкий Recall (0.65) является осознанным компромиссом для обеспечения комфортного пользовательского опыта.

## 📚 Использование

```python
from toxicity_detector import ToxicityDetector

# Создать детектор (можно указывать устройство)
detector = ToxicityDetector(device="cpu")

texts = [
    'Ты берега попутал?',                        # {'is_toxic': False, 'score': 0.2875}
    'Это правый берег реки, не путай с левым.',  # {'is_toxic': False, 'score': 0.0165}
    "Ты дуралей."                                # {'is_toxic': True, 'score': 0.9969}
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

Распространяется по лицензии **[MIT](LICENSE.txt)**.

Проект использует компоненты с открытым исходным кодом. Сведения о лицензиях см. в **[pyproject.toml](pyproject.toml)** и на официальных ресурсах зависимостей.
