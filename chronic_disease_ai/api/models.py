from django.db import models

class Patient(models.Model):
    name = models.CharField(max_length=200)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    smoking = models.BooleanField(default=False)
    family_history = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class ClinicalRecord(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    glucose = models.FloatField()
    blood_pressure = models.FloatField()
    cholesterol = models.FloatField()
    physical_activity = models.IntegerField()  # minutos por día
    bmi = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)


class Prediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    disease = models.CharField(max_length=100)  # "diabetes", "hipertension", etc.
    probability = models.FloatField()
    model_version = models.CharField(max_length=50)

    created_at = models.DateTimeField(auto_now_add=True)


class TrainingSession(models.Model):
    dataset_name = models.CharField(max_length=200)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)
