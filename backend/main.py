from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .api.router import api_router
from .config import settings
from .services.inference_engine import get_inference_engine
from .services.treatment_service import get_treatment_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print(f"Starting {settings.PROJECT_NAME} API v{settings.VERSION}")
    print("=" * 60)
    treatment_svc = get_treatment_service()
    engine = get_inference_engine()
    crops = treatment_svc.repository.list_supported_crops()
    print(f"[OK] Inference Engine: {engine.engine_name}")
    print(f"[OK] Treatment Database loaded: {len(crops)} crops, {len(treatment_svc.repository.list_all_canonical_classes())} classes")
    print(f"[OK] Supported crops: {', '.join(crops)}")
    print("=" * 60)
    yield
    print(f"Shutting down {settings.PROJECT_NAME} API...")


app = FastAPI(
    title=f"{settings.PROJECT_NAME} - Backend API",
    version=settings.VERSION,
    description=(
        "Production-ready AI Plant Health Assistant API. "
        "Provides disease classification, plant part detection, out-of-scope/uncertainty rejection, "
        "and Integrated Pest Management (IPM) treatment recommendations."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": "HTTP_EXCEPTION",
            "message": str(exc.detail),
            "hint": "Check request parameters and try again.",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_msg = "; ".join([f"{e['loc'][-1]}: {e['msg']}" for e in errors]) if errors else "Validation failed"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "message": error_msg,
            "hint": "Check request body/form field formats against the API schema.",
            "details": exc.errors(),
        },
    )


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/app", response_class=FileResponse, tags=["Web UI"])
@app.get("/app/home", response_class=FileResponse, tags=["Web UI"])
@app.get("/app/workflow", response_class=FileResponse, tags=["Web UI"])
async def serve_home():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Home page not found")


@app.get("/app/scan", response_class=FileResponse, tags=["Web UI"])
async def serve_scan():
    page = static_dir / "scan.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Scan page not found")


@app.get("/app/overview", response_class=FileResponse, tags=["Web UI"])
async def serve_overview():
    page = static_dir / "overview.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Overview page not found")


@app.get("/app/history", response_class=FileResponse, tags=["Web UI"])
async def serve_history():
    page = static_dir / "history.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="History page not found")


@app.get("/app/insights", response_class=FileResponse, tags=["Web UI"])
async def serve_insights():
    page = static_dir / "insights.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Disease Insights page not found")


@app.get("/app/settings", response_class=FileResponse, tags=["Web UI"])
async def serve_settings():
    page = static_dir / "settings.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Settings page not found")


@app.get("/", tags=["General"])
async def root(request: Request):
    accept = request.headers.get("accept", "")
    index_file = static_dir / "index.html"
    if "text/html" in accept and not "application/json" in accept and index_file.exists():
        return FileResponse(index_file)
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "predict_endpoint": "/api/v1/predict",
        "web_ui": "/app",
    }


app.include_router(api_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
