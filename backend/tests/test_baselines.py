"""
backend/tests/test_baselines.py

Tests for the deterministic change_map and optical_sar baseline modules.
Uses only synthetic GeoTIFF fixtures — no real satellite data.
"""
import io
import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_origin
    from backend.app.change_map import compute_change_baseline
    from backend.app.optical_sar import verify_optical_sar_pair
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def _make_geotiff(value: int = 100, width: int = 10, height: int = 10) -> bytes:
    """Creates a minimal single-band GeoTIFF filled with a constant value."""
    if not RASTERIO_AVAILABLE:
        return b"fake_tiff"
    transform = from_origin(0, 0, 1, 1)
    data = np.full((1, height, width), value, dtype=np.uint8)
    buf = io.BytesIO()
    with rasterio.open(
        buf, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=data.dtype,
        crs='EPSG:4326', transform=transform,
    ) as dst:
        dst.write(data)
    return buf.getvalue()


def _make_geotiff_no_crs(value: int = 100) -> bytes:
    """Creates a GeoTIFF without a CRS."""
    if not RASTERIO_AVAILABLE:
        return b"fake_tiff"
    transform = from_origin(0, 0, 1, 1)
    data = np.full((1, 10, 10), value, dtype=np.uint8)
    buf = io.BytesIO()
    with rasterio.open(
        buf, 'w', driver='GTiff',
        height=10, width=10, count=1, dtype=data.dtype,
        transform=transform,
    ) as dst:
        dst.write(data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# change_map tests
# ---------------------------------------------------------------------------

class TestChangeMap:
    def test_identical_images_no_change(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img = _make_geotiff(100)
        result = compute_change_baseline(img, img)
        assert result["status"] == "SUCCESS"
        assert "No significant change" in result["answer"]

    def test_different_images_change_detected(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img1 = _make_geotiff(0)
        img2 = _make_geotiff(255)
        result = compute_change_baseline(img1, img2)
        assert result["status"] == "SUCCESS"
        assert "change" in result["answer"].lower()
        assert result["confidence"] is None

    def test_change_mask_visual_output_is_base64(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img1 = _make_geotiff(0)
        img2 = _make_geotiff(200)
        result = compute_change_baseline(img1, img2)
        assert result.get("visual_output", "").startswith("data:image/png;base64,")

    def test_incompatible_dimensions_returns_invalid(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img1 = _make_geotiff(100, width=10, height=10)
        img2 = _make_geotiff(100, width=20, height=20)
        result = compute_change_baseline(img1, img2)
        assert result["status"] == "INVALID_INPUT"

    def test_provenance_not_falsely_adapted(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img = _make_geotiff(100)
        result = compute_change_baseline(img, img)
        assert result["provenance"]["remote_sensing_adapted"] is False

    def test_corrupted_bytes_returns_error(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        result = compute_change_baseline(b"not_a_tiff", b"not_a_tiff")
        assert result["status"] in ("ERROR", "INVALID_INPUT")


# ---------------------------------------------------------------------------
# optical_sar tests
# ---------------------------------------------------------------------------

class TestOpticalSar:
    def test_compatible_pair_succeeds(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img = _make_geotiff(100)
        result = verify_optical_sar_pair(img, img)
        assert result["status"] == "SUCCESS"
        assert len(result["evidence"]) > 0

    def test_incompatible_dimensions_rejected(self):
        if not RASTERIO_AVAILABLE: pytest.skip("Rasterio unavailable")
        img1 = _make_geotiff(100, width=10, height=10)
        img2 = _make_geotiff(100, width=20, height=20)
        result = verify_optical_sar_pair(img1, img2)
        assert result["status"] == "INVALID_INPUT"
        assert "dimension" in result["answer"].lower()

    def test_provenance_not_falsely_adapted(self):
        img = _make_geotiff(100)
        result = verify_optical_sar_pair(img, img)
        assert result["provenance"]["remote_sensing_adapted"] is False

    def test_modalities_labeled_correctly(self):
        img = _make_geotiff(100)
        result = verify_optical_sar_pair(img, img)
        assert result["provenance"]["input_modalities"] == ["optical", "sar"]

    def test_corrupted_bytes_returns_error(self):
        result = verify_optical_sar_pair(b"garbage", b"garbage")
        assert result["status"] in ("ERROR", "INVALID_INPUT")
