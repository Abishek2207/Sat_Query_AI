import io
import os
import rasterio
from PIL import Image
from rasterio.io import MemoryFile

from .schemas import ValidationResult

def is_safe_path(filename: str) -> bool:
    if ".." in filename or "/" in filename or "\\" in filename:
        return False
    return True

def validate_image_bytes(
    filename: str,
    file_bytes: bytes,
    benchmark_mode: bool = False,
) -> ValidationResult:

    if not is_safe_path(filename):
        return ValidationResult(valid=False, reason="Path traversal detected in filename.")

    MAX_SIZE = 100 * 1024 * 1024 # 100MB
    if len(file_bytes) > MAX_SIZE:
        return ValidationResult(valid=False, reason="File exceeds maximum size of 100MB.")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in {"tif", "tiff", "geotiff"}:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    tags = src.tags()
                    acquisition_time = tags.get("TIFFTAG_DATETIME") or tags.get("ACQUISITION_DATETIME")
                    
                    transform = None
                    if src.transform:
                        transform = [
                            src.transform.a, src.transform.b, src.transform.c,
                            src.transform.d, src.transform.e, src.transform.f,
                        ]

                    modality = "optical"
                    if "sar" in filename.lower() or "s1" in filename.lower() or "radar" in filename.lower():
                        modality = "sar"
                    elif "optical" in filename.lower() or "s2" in filename.lower():
                        modality = "optical"

                    return ValidationResult(
                        valid=True,
                        crs=src.crs.to_string() if src.crs else None,
                        transform=transform,
                        width=src.width,
                        height=src.height,
                        bands=src.count,
                        modality=modality,
                        acquisition_time=acquisition_time,
                        file_format="geotiff",
                    )
        except Exception as exc:
            return ValidationResult(valid=False, reason=f"Rasterio decode failed: {exc}")

    if ext in {"png", "jpg", "jpeg"}:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()
            image = Image.open(io.BytesIO(file_bytes))
            
            modality = "optical"
            if "sar" in filename.lower() or "radar" in filename.lower():
                modality = "sar"
                
            return ValidationResult(
                valid=True,
                width=image.width,
                height=image.height,
                modality=modality,
                file_format="benchmark_rgb",
            )
        except Exception:
            return ValidationResult(valid=False, reason="Invalid PNG/JPEG image structure.")

    return ValidationResult(
        valid=False,
        reason="Unsupported file extension. Supported formats: GeoTIFF/TIFF and benchmark PNG/JPEG."
    )
