from django.contrib import admin

from .models import Patient, ClinicalRecord, Prediction, TrainingSession

admin.site.register(Patient)
admin.site.register(ClinicalRecord)
admin.site.register(Prediction)
admin.site.register(TrainingSession)

