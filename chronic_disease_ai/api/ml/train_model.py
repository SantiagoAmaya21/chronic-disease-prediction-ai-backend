import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")

def train_model():
    print(" Starting training...")

    # Cargar dataset
    df = pd.read_csv(DATASET_PATH)
    print(f" Dataset loaded: {len(df)} rows")

    # Separar features y target
    feature_cols = ["age", "glucose", "blood_pressure", "cholesterol", "bmi", "physical_activity"]
    X = df[feature_cols]
    y = df["label"]

    # Análisis de balance de clases
    print(f"\n  Class distribution (original):")
    class_counts = Counter(y)
    for label, count in class_counts.items():
        print(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split ANTES de balancear (importante)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25,  # Aumentado a 25% para test más robusto
        random_state=42,
        stratify=y
    )

    print(f"\n Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # BALANCEAR CLASES solo en train (con SMOTE)
    print("\n Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"   After SMOTE:")
    balanced_counts = Counter(y_train_balanced)
    for label, count in balanced_counts.items():
        print(f"     Class {label}: {count}")

    # MODELO MUY RESTRICTIVO para evitar overfitting
    model = RandomForestClassifier(
        n_estimators=50,          # Menos árboles
        max_depth=4,              # Árboles poco profundos
        min_samples_split=20,     # Requiere muchos datos para dividir
        min_samples_leaf=10,      # Hojas grandes
        max_features='sqrt',      # Pocas features por árbol
        max_leaf_nodes=15,        # Limita nodos totales
        random_state=42,
        n_jobs=-1
    )

    # Validación cruzada
    print("\n Cross-validation (5-fold) on TRAIN set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='f1')
    print(f"   CV F1 scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Entrenar modelo final
    model.fit(X_train_balanced, y_train_balanced)
    print(" Model trained!")

    # Evaluar en TRAIN (balanceado)
    y_train_pred = model.predict(X_train_balanced)
    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    train_f1 = f1_score(y_train_balanced, y_train_pred)

    # Evaluar en TEST (desbalanceado original)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n TRAINING METRICS (balanced data):")
    print(f"   Accuracy: {train_accuracy:.4f}")
    print(f"   F1: {train_f1:.4f}")
    
    print(f"\n TEST METRICS (original unbalanced data):")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   F1: {test_f1:.4f}")
    print(f"   AUC-ROC: {test_auc:.4f}")

    # Detección de overfitting
    overfitting_gap = train_accuracy - test_accuracy
    print(f"\n Overfitting check:")
    print(f"   Gap (train - test): {overfitting_gap:.4f}")
    if overfitting_gap > 0.15:
        print("OVERFITTING DETECTED!")
    elif overfitting_gap < -0.05:
        print("UNDERFITTING?")
    else:
        print("Acceptable generalization")

    # Reporte detallado
    print(f"\n Classification Report (TEST):")
    print(classification_report(y_test, y_test_pred, target_names=['No Disease', 'Disease']))

    print(f"\n Confusion Matrix (TEST):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predicted")
    print(f"                 0    1")
    print(f"   Actual  0   {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"           1   {cm[1,0]:3d}  {cm[1,1]:3d}")

    # Feature importance
    print(f"\n Feature Importance:")
    importances = sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"   {feat:20s}: {imp:.4f} {bar}")

    # Guardar modelo y scaler
    model_path = os.path.join(current_dir, "model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n Model saved to: {model_path}")
    print(f" Scaler saved to: {scaler_path}")

    # IMPORTANTE: Retornar métricas de TEST
    return test_accuracy, test_precision, test_recall, test_f1