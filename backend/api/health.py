from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends
from ..config import settings
from ..schemas.common import HealthCheckResponse, InfoResponse
from ..services.inference_engine import BaseModelEngine, get_inference_engine
from ..services.treatment_service import TreatmentService, get_treatment_service

router = APIRouter(tags=["System & Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System Health Check",
    description="Check backend API operational status, active model inference engine, and treatment knowledge base.",
)
async def health_check(
    engine: BaseModelEngine = Depends(get_inference_engine),
    treatment_svc: TreatmentService = Depends(get_treatment_service),
) -> HealthCheckResponse:
    crops = treatment_svc.repository.list_supported_crops()
    return HealthCheckResponse(
        status="healthy",
        service="PlantScan AI API",
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        active_model_engine=engine.engine_name,
        treatment_database_loaded=len(crops) > 0,
        supported_crops_count=len(crops),
    )


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="API & Model Metadata",
    description="Retrieve project information, supported crop species, confidence thresholds, and image upload constraints.",
)
async def api_info(
    treatment_svc: TreatmentService = Depends(get_treatment_service),
) -> InfoResponse:
    return InfoResponse(
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="AI-powered plant disease diagnosis and sustainable Integrated Pest Management advisory.",
        supported_crops=treatment_svc.repository.list_supported_crops(),
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        max_image_size_mb=settings.MAX_IMAGE_SIZE_MB,
        allowed_formats=sorted(list(settings.ALLOWED_EXTENSIONS)),
    )
