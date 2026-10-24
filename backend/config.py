from pathlib import Path
from typing import List, Set, Tuple


class Settings:
    PROJECT_NAME: str = "PlantScan AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Thresholds
    CONFIDENCE_THRESHOLD: float = 0.70
    
    # Image constraints
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
    MIN_IMAGE_DIMENSIONS: Tuple[int, int] = (32, 32)
    MAX_IMAGE_DIMENSIONS: Tuple[int, int] = (4096, 4096)
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ALLOWED_MIME_TYPES: Set[str] = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "application/octet-stream",
    }
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    TREATMENT_DB_DIR: Path = PROJECT_ROOT / "treatment_database"
    TAXONOMY_PATH: Path = PROJECT_ROOT / "datasets" / "reports" / "model" / "final_class_taxonomy_v2.csv"
    WEIGHTS_DIR: Path = PROJECT_ROOT / "ml" / "weights"
    
    # Supported crops list
    SUPPORTED_CROPS: List[str] = [
        "Apple",
        "Banana",
        "Bell pepper",
        "Cotton",
        "Grape",
        "Maize",
        "Mango",
        "Potato",
        "Rice",
        "Tomato",
    ]

    # Model configuration
    DEFAULT_MODEL_NAME: str = "MobileNetV2-PlantScan-v1"
    IMAGE_INPUT_SIZE: Tuple[int, int] = (224, 224)
    ENABLE_MOCK_FALLBACK: bool = True

    # Security & CORS configuration
    ENVIRONMENT: str = "development"
    ENABLE_SIMULATION_HEADERS: bool = True
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False


settings = Settings()
