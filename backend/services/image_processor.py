import base64
import io
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException, status

from ..config import settings


class ImageValidationError(HTTPException):
    def __init__(self, message: str, hint: Optional[str] = None, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            detail={
                "status": "invalid_image",
                "error_code": "INVALID_IMAGE",
                "message": message,
                "hint": hint or "Please upload a valid JPEG, PNG, or WebP image under 10MB.",
            },
        )


class ImageProcessor:
    """
    Validates, sanitizes, and prepares raw image bytes or base64 strings
    for machine learning inference.
    """

    @staticmethod
    def validate_and_load_bytes(image_bytes: bytes, filename: Optional[str] = None) -> Image.Image:
        if not image_bytes:
            raise ImageValidationError(
                message="Uploaded file is empty (0 bytes).",
                hint="Please select a valid image file from your device.",
            )

        if len(image_bytes) > settings.MAX_IMAGE_SIZE_BYTES:
            raise ImageValidationError(
                message=f"Image file size ({len(image_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed {settings.MAX_IMAGE_SIZE_MB} MB.",
                hint=f"Compress or resize the image to under {settings.MAX_IMAGE_SIZE_MB} MB before uploading.",
            )

        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in settings.ALLOWED_EXTENSIONS:
                raise ImageValidationError(
                    message=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}.",
                    hint="Convert your image to JPEG or PNG before uploading.",
                )

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except UnidentifiedImageError:
            raise ImageValidationError(
                message="The uploaded file is not a recognized image or is severely corrupted.",
                hint="Ensure you are uploading a valid JPEG, PNG, BMP, or WebP image.",
            )
        except Exception as e:
            raise ImageValidationError(
                message=f"Failed to process image: {str(e)}",
                hint="The image file appears corrupted or truncated. Please take a new photo.",
            )

        width, height = image.size
        min_w, min_h = settings.MIN_IMAGE_DIMENSIONS
        max_w, max_h = settings.MAX_IMAGE_DIMENSIONS

        if width < min_w or height < min_h:
            raise ImageValidationError(
                message=f"Image dimensions ({width}x{height}) are too small. Minimum required is {min_w}x{min_h} pixels.",
                hint="Upload a higher resolution photo for accurate plant disease detection.",
            )

        if width > max_w or height > max_h:
            raise ImageValidationError(
                message=f"Image dimensions ({width}x{height}) exceed maximum allowed {max_w}x{max_h} pixels.",
                hint="Downscale or crop the photo before sending.",
            )

        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @classmethod
    def load_from_base64(cls, base64_str: str) -> Image.Image:
        if not base64_str or not base64_str.strip():
            raise ImageValidationError(
                message="Base64 image string is empty.",
                hint="Provide a valid Base64 encoded image string.",
            )

        clean_str = base64_str.strip()
        if "," in clean_str and ("data:image" in clean_str or "base64" in clean_str):
            clean_str = clean_str.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(clean_str)
        except Exception as e:
            raise ImageValidationError(
                message=f"Failed to decode Base64 string: {str(e)}",
                hint="Verify that your Base64 string is correctly formatted.",
            )

        return cls.validate_and_load_bytes(image_bytes)

    @staticmethod
    def preprocess_for_model(
        image: Image.Image,
        target_size: Tuple[int, int] = settings.IMAGE_INPUT_SIZE,
    ) -> np.ndarray:
        resized = image.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        return np.expand_dims(arr, axis=0)
