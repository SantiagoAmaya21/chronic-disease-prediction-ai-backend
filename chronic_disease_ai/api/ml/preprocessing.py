import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_input(data):
    df_input = pd.DataFrame([{
        "age": data["age"],
        "glucose": data["glucose"],
        "blood_pressure": data["blood_pressure"],
        "cholesterol": data["cholesterol"],
        "bmi": data["bmi"],
        "physical_activity": data["physical_activity"]
    }])

    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    df_scaled = scaler.transform(df_input)

    return df_scaled

def make_prediction(data):
    model_path = os.path.join(BASE_DIR, "model.pkl")
    model = joblib.load(model_path)

    X = preprocess_input(data)
    # usar predict_proba para obtener la probabilidad de clase 1
    prob = model.predict_proba(X)[0][1]
    return prob