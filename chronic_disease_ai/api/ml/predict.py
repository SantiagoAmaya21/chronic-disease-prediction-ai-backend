import joblib
import os
from .preprocessing import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def make_prediction(data):
    model_path = os.path.join(BASE_DIR, "model.pkl")
    model = joblib.load(model_path)

    processed = preprocess_input(data)

    prob = model.predict_proba(processed)[0][1]   # probabilidad de enfermedad
    return float(prob)
