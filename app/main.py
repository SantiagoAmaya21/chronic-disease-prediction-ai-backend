from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(
    title="Chronic Disease Prediction API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Backend is running!"}

app.include_router(api_router, prefix="/api")

