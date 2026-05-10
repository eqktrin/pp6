import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Загрузка данных
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
feature_names = joblib.load("data/processed/feature_names.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric='auc'
)

print("Обучение XGBoost...")
model.fit(X_train, y_train)

# Оценка
preds = model.predict(X_test)
print("\n" + "="*50)
print(classification_report(y_test, preds, target_names=['Норма', 'Аномалия']))
print("="*50)

# Feature importance
importance = model.feature_importances_
top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:15]

print("\nТоп-15 важных признаков:")
for name, imp in top_features:
    print(f"{name:25} {imp:.4f}")

# Сохранение отчёта
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
plt.savefig("outputs/reports/confusion_matrix.png")