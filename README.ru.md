# ru-toxicity-detector

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/Лицензия-PolyForm-blue) [![Sponsor](https://img.shields.io/badge/Поддержать-%E2%9D%A4-red)](https://kvaytg.ru/donate.php?lang=ru)

Определение токсичности русскоязычного текста с помощью модели [KvaytG/rubert-tiny2-toxic-detector](https://huggingface.co/KvaytG/rubert-tiny2-toxic-detector).

## Как это работает

Пакет напрямую использует предварительно обученную модель классификации текста через Hugging Face Transformers.

Порог токсичности по умолчанию — `0.70`. Текст считается токсичным, если рассчитанная вероятность строго больше заданного порога. Поле `score` содержит вероятность, округлённую до четырёх знаков после запятой.

При первом создании детектора модель и токенизатор загружаются с Hugging Face и затем кэшируются. При первой загрузке требуется интернет, если только `model_name` не указывает на локальную копию модели.

## Установка

```bash
pip install git+https://github.com/KvaytG/ru-toxicity-detector.git
```

Для локальной разработки:

```bash
pip install -e ".[dev]"
```

## Использование

```python
from toxicity_detector import ToxicityDetector

# Создать детектор (можно указывать устройство)
detector = ToxicityDetector(device="cpu")

texts = [
    "Я люблю тебя",  # {'is_toxic': False, 'score': 0.0119}
    "Пошёл нахуй"    # {'is_toxic': True,  'score': 0.9736}
]

# Предсказания по одному тексту
for idx, text in enumerate(texts, start=1):
    print(f"{idx}) {detector.predict(text)}")

# Предсказания батчем (быстрее для нескольких текстов)
results = detector.predict_batch(texts)
for idx, res in enumerate(results, start=1):
    print(f"{idx}) {res}")
```

## Порог и устройство

```python
# Более высокий порог может сократить ложные срабатывания,
# а более низкий — повысить чувствительность.
detector = ToxicityDetector(threshold=0.80, device="cpu")

# При наличии совместимой версии PyTorch с поддержкой CUDA:
detector = ToxicityDetector(device="cuda")
```

Параметры конструктора:

| Параметр     | По умолчанию                           | Описание                                              |
|--------------|----------------------------------------|-------------------------------------------------------|
| `threshold`  | `0.70`                                 | Порог вероятности токсичности от `0.0` до `1.0`       |
| `device`     | `"cpu"`                                | Устройство PyTorch для инференса                      |
| `model_name` | `"KvaytG/rubert-tiny2-toxic-detector"` | ID модели на Hugging Face или путь к локальной модели |
| `max_length` | `128`                                  | Максимальная длина последовательности токенов         |
| `batch_size` | `64`                                   | Количество текстов в одном пакете инференса           |

Методы `predict(text)` и `predict_batch(texts)` возвращают словари с одинаковыми полями:

- `is_toxic` — превышает ли вероятность заданный порог.
- `score` — вероятность токсичности, округлённая до четырёх знаков после запятой.

## Модель и метрики

Информация об обучении, результатах оценки, рекомендуемом пороге и лицензии Apache-2.0 приведена в [карточке модели](https://huggingface.co/KvaytG/rubert-tiny2-toxic-detector).

## Лицензия проекта

Исходный код проекта распространяется по лицензии [PolyForm Noncommercial](LICENSE.md). Для предобученной модели действуют её собственная лицензия и условия использования.
