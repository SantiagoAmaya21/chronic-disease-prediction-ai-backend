from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Chronic Disease Prediction API"
    VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()
