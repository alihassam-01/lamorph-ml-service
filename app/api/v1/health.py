from fastapi import APIRouter
from app.models.manager import model_manager

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "wrinkle_model": model_manager.wrinkle_session is not None,
            "eyebag_model": model_manager.eyebag_session is not None
        }
    }
