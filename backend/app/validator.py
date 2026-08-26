import io

import rasterio
from PIL import Image
from rasterio.io import MemoryFile

from .schemas import ValidationResult


def validate_image_bytes(
    filename: str,
    file_bytes: bytes,
    benchmark_mode: bool = False,
) -> ValidationResult:

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in {"tif", "tiff", "geotiff"}:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:

                    tags = src.tags()

                    acquisition_time = (
                        tags.get("TIFFTAG_DATETIME")
                        or tags.get("ACQUISITION_DATETIME")
                    )

                    transform = None

                    if src.transform:
                        transform = [
                            src.transform.a,
                            src.transform.b,
                            src.transform.c,
                            src.transform.d,
                            src.transform.e,
                            src.transform.f,
                        ]

                    return ValidationResult(
                        valid=True,
                        crs=src.crs.to_string() if src.crs else None,
                        transform=transform,
                        width=src.width,
                        height=src.height,
                        bands=src.count,
                        modality="unknown",
                        acquisition_time=acquisition_time,
                        file_format="geotiff",
                    )

        except Exception as exc:
            return ValidationResult(
                valid=False,
                reason=f"Rasterio decode failed: {exc}",
            )

    if ext in {"png", "jpg", "jpeg"}:

        if not benchmark_mode:
            return ValidationResult(
                valid=False,
                reason=(
                    "PNG/JPEG rejected. "
                    "Only GeoTIFF/TIFF is allowed outside benchmark mode."
                ),
            )

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()

            image = Image.open(io.BytesIO(file_bytes))

            return ValidationResult(
                valid=True,
                width=image.width,
                height=image.height,
                modality="unknown",
                file_format="benchmark_rgb",
            )

        except Exception:
            return ValidationResult(
                valid=False,
                reason="Invalid PNG/JPEG image structure.",
            )

    return ValidationResult(
        valid=False,
        reason=(
            "Unsupported file extension. "
            "Supported formats: GeoTIFF/TIFF and benchmark PNG/JPEG."
        ),
    )
