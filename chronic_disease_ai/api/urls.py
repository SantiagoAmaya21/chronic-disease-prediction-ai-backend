from django.urls import path
from .views import create_patient, create_clinical_record, train, predict

urlpatterns = [
    path("patient/", create_patient),
    path("clinical/", create_clinical_record),
    path("train/", train),
    path("predict/", predict),
]
