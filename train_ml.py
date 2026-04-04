import pandas as pa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

df = pa.read_json(DATA_DIR / "dataset_fps_fr_labeled.json")
print(f"Dataset : {df.shape[0]} vidéos")

#Features pré-publication uniquement
features = ["title_length", "has_number", "has_punctuation", "upper_word_count", "day_of_week", "hour", "duration_seconds", "subscriber_count"]

X = df[features]
y = df["label"]

le = LabelEncoder()
y = le.fit_transform(y)

target_names = ["moderate", "strong", "viral", "weak"]

# Split 70% train / 15% validation / 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train : {X_train.shape[0]}")
print(f"Validation : {X_val.shape[0]}")
print(f"Test : {X_test.shape[0]}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_val = rf.predict(X_val)

print(f"Accuracy : {accuracy_score(y_val, y_pred_val):.2%}")
print(classification_report(y_val, y_pred_val, target_names = target_names))

importances = pa.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Importance des features :")
print(importances.to_string())  

 # XGBOOST
xg = XGBClassifier(n_estimators=100, random_state=42)
xg.fit(X_train, y_train)
y_pred_val = xg.predict(X_val)

print(f"Accuracy XGBOOST : {accuracy_score(y_val, y_pred_val): .2%}")
print(classification_report(y_val, y_pred_val, target_names = target_names))