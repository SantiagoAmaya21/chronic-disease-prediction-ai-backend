from django.shortcuts import render

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

from .models import Patient, ClinicalRecord, Prediction, TrainingSession
from .serializers import PatientSerializer, ClinicalRecordSerializer, PredictionSerializer, TrainingSessionSerializer

from .ml.train_model import train_model
from .ml.predict import make_prediction


@api_view(["POST"])
def create_patient(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=400)


@api_view(["POST"])
def create_clinical_record(request):
    serializer = ClinicalRecordSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=400)


@api_view(["POST"])
def train(request):
    """
    Entrena el modelo usando el dataset en api/dataset/dataset.csv
    """
    accuracy, precision, recall, f1 = train_model()

    ts = TrainingSession.objects.create(
        dataset_name="dataset.csv",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )

    return Response(TrainingSessionSerializer(ts).data)


@api_view(["POST"])
def predict(request):
    data = {k: float(v) for k, v in request.data.items()}

    prob = make_prediction(data)

    result = {
        "probability": prob,
        "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    }

    return Response(result)
