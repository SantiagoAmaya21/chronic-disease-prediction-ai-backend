from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_patients():
    return {"message": "Patients endpoint working!"}
