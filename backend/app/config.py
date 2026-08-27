import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATASETS_DIR: str = os.getenv("DATASETS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets")))
    MODELS_DIR: str = os.getenv("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models")))
    VRAM_MIN_REQ_MB: int = int(os.getenv("VRAM_MIN_REQ_MB", "4000"))
    
    class Config:
        env_file = ".env"

settings = Settings()
