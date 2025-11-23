import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from django.conf import settings


# Carpeta donde está este script: api/ml
current_dir = os.path.dirname(os.path.abspath(__file__))

# Subimos 2 niveles hasta la raíz del proyecto
BASE_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Ruta correcta al CSV
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")

print("Ruta dataset:", DATASET_PATH)  # para verificar

def train_model():
    print(" Starting training...")

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} rows")

    X = df[["age", "glucose", "blood_pressure", "cholesterol", "bmi", "physical_activity"]]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Metrics -> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    print("Model and scaler saved!")

    return accuracy, precision, recall, f1
