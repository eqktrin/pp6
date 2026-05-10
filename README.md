# RBC Feature Extractor for CARCINOMA-RANK

Модуль извлечения признаков из клеток эритроцитов (RBC) для автоматизированного ранжирования аномалий.  

## Возможности

- Извлечение **22 признаков** из клетки: геометрические, цветовые, текстурные (GLCM), гистограмма яркости.
- Поддержка датасета **Elsafty RBCs for Classification** (240k клеток, 9 классов).
- Модульная структура: легко добавлять новые признаки или заменять модели.
- Baseline модель (XGBoost) с метриками Precision / Recall ≈ 0.99.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/АРТЕМ_АНДРЕЙ_ХЗ_КАКИЕ_У_ВАС_ЛОГИНЫ/pp6.git
   cd pp6

## Датасет

Используется датасет **Elsafty RBCs for Classification** (240 507 клеток, 9 классов).  

Ссылка: [https://figshare.com/articles/dataset/Elsafty_RBCs_for_Classification/25599981](https://figshare.com/articles/dataset/Elsafty_RBCs_for_Classification/25599981)


## Запустите извлечение признаков:
  ```bash
  python scripts/build_features.py
  ```



## Обучите baseline-модель:
  ```bash
  python scripts/train_baseline.py
  ```

## Файл вместе с кодом экстрактора.

Этот файл содержит всё, что нужно: В нём хранятся все обученные "деревья решений", важнейшие параметры модели и её конфигурация.

Для загрузки на стороне:

  ```python
  import xgboost as xgb
  model = xgb.XGBClassifier()
  model.load_model("xgb_anomaly_model.json")
