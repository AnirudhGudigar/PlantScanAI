from fastapi import APIRouter
from .health import router as health_router
from .predict import router as predict_router
from .treatments import router as treatments_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(predict_router)
api_router.include_router(treatments_router)
