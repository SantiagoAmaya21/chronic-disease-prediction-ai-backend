from fastapi import APIRouter
from .endpoints import patients

api_router = APIRouter()

api_router.include_router(
    patients.router,
    prefix="/patients",
    tags=["Patients"]
)
