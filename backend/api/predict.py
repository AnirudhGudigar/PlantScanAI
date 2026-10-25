import time
from typing import Optional
# pyrefly: ignore [missing-import]
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..schemas.common import ErrorDetail, PredictionStatus
from ..schemas.prediction import Base64PredictRequest, PredictionResponse
from ..services.image_processor import ImageProcessor, ImageValidationError
from ..services.inference_engine import BaseModelEngine, get_inference_engine
from ..services.treatment_service import TreatmentService, get_treatment_service

router = APIRouter(tags=["Disease Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Diagnose Plant Disease from Image Upload",
    description=(
        "Upload a plant leaf/crop image (JPEG, PNG, WebP) to classify disease, identify plant part, "
        "calculate Plant Health Score, and obtain sustainable Integrated Pest Management (IPM) recommendations."
    ),
    responses={
        400: {"model": ErrorDetail, "description": "Invalid image format, corrupted file, or bad dimensions"},
        422: {"description": "Unprocessable entity / validation error"},
        500: {"model": ErrorDetail, "description": "Internal server or inference failure"},
    },
)
async def predict_image(
    file: UploadFile = File(..., description="Plant image file (JPEG, PNG, WebP, max 10MB)"),
    crop_hint: Optional[str] = Form(None, description="Optional target crop hint (e.g., 'Tomato', 'Rice', 'Mango')"),
    plant_part_hint: Optional[str] = Form(None, description="Optional plant part hint ('Leaf', 'Stem', 'Fruit', 'Bark')"),
    language: Optional[str] = Form("en", description="Preferred response language code ('en', 'kn', 'hi')"),
    x_simulate_class: Optional[str] = Header(None, alias="X-Simulate-Class", description="Test override: forced class"),
    x_simulate_confidence: Optional[float] = Header(None, alias="X-Simulate-Confidence", description="Test override: forced confidence"),
    x_simulate_crop: Optional[str] = Header(None, alias="X-Simulate-Crop", description="Test override: forced crop"),
    x_simulate_status: Optional[str] = Header(None, alias="X-Simulate-Status", description="Test override: forced status"),
    engine: BaseModelEngine = Depends(get_inference_engine),
    treatment_svc: TreatmentService = Depends(get_treatment_service),
) -> PredictionResponse:
    start_time = time.perf_counter()

    try:
        image_bytes = await file.read()
        pil_image = await run_in_threadpool(
            ImageProcessor.validate_and_load_bytes, image_bytes, filename=file.filename
        )
    except ImageValidationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "invalid_image",
                "error_code": "FILE_READ_ERROR",
                "message": f"Could not process uploaded file: {str(e)}",
                "hint": "Ensure the uploaded file is an uncorrupted image.",
            },
        )

    input_array = await run_in_threadpool(ImageProcessor.preprocess_for_model, pil_image)

    force_sim = None
    if settings.ENABLE_SIMULATION_HEADERS:
        if x_simulate_class or x_simulate_confidence is not None or x_simulate_crop:
            force_sim = {
                "class": x_simulate_class or "tomato_early_blight",
                "confidence": x_simulate_confidence if x_simulate_confidence is not None else 0.94,
                "crop": x_simulate_crop or "Tomato",
            }

    try:
        prediction = await run_in_threadpool(
            engine.predict,
            image_array=input_array,
            crop_hint=crop_hint,
            plant_part_hint=plant_part_hint,
            force_simulation=force_sim,
            original_image=pil_image,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error_code": "INFERENCE_FAILED",
                "message": f"AI model inference failed: {str(e)}",
                "hint": "Please try again later or contact system administration.",
            },
        )

    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    forced_st = (
        PredictionStatus(x_simulate_status)
        if (settings.ENABLE_SIMULATION_HEADERS and x_simulate_status)
        else None
    )
    response = await run_in_threadpool(
        treatment_svc.build_prediction_response,
        prediction=prediction,
        inference_time_ms=inference_time_ms,
        forced_status=forced_st,
        language=language,
    )

    return response


@router.post(
    "/predict/json",
    response_model=PredictionResponse,
    summary="Diagnose Plant Disease from Base64 Image JSON",
    description="Send a Base64-encoded image in a JSON payload for environments where multipart uploads are inconvenient.",
    responses={
        400: {"model": ErrorDetail, "description": "Invalid Base64 string or invalid image"},
    },
)
async def predict_base64_json(
    request: Base64PredictRequest,
    engine: BaseModelEngine = Depends(get_inference_engine),
    treatment_svc: TreatmentService = Depends(get_treatment_service),
) -> PredictionResponse:
    start_time = time.perf_counter()
    pil_image = await run_in_threadpool(ImageProcessor.load_from_base64, request.image_base64)
    input_array = await run_in_threadpool(ImageProcessor.preprocess_for_model, pil_image)

    prediction = await run_in_threadpool(
        engine.predict,
        image_array=input_array,
        crop_hint=request.crop_hint,
        plant_part_hint=request.plant_part_hint,
        original_image=pil_image,
    )

    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return await run_in_threadpool(
        treatment_svc.build_prediction_response,
        prediction=prediction,
        inference_time_ms=inference_time_ms,
        language=request.language,
    )
