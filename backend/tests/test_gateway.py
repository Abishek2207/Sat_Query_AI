import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health").json()
    assert res["api_status"] == "AVAILABLE"

def test_png_rejected_without_benchmark(valid_png_bytes):
    files = [("files", ("test.png", valid_png_bytes, "image/png"))]
    res = client.post("/analyze", data={"query": "test"}, files=files).json()
    assert res["status"] == "INVALID_INPUT"
    assert "PNG/JPEG rejected" in res["execution_trace"][0]

def test_png_accepted_with_benchmark(valid_png_bytes):
    files = [("files", ("test.png", valid_png_bytes, "image/png"))]
    res = client.post("/analyze", data={"query": "test", "benchmark_mode": "true"}, files=files).json()
    assert res["status"] in ["SUCCESS", "MODEL_UNAVAILABLE"]

def test_synthetic_geotiff_accepted(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "test"}, files=files).json()
    assert res["status"] in ["SUCCESS", "MODEL_UNAVAILABLE"]
    assert res["validation"][0]["crs"] == "EPSG:4326"

def test_vqa_routing(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "How many buildings are visible?"}, files=files).json()
    assert res["task"] == "vqa"

def test_land_cover_routing(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "What is the dominant land cover?"}, files=files).json()
    assert res["task"] == "land_cover_classification"

def test_grounding_routing(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Highlight the water"}, files=files).json()
    assert res["task"] == "grounding"

def test_captioning_routing(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Describe the scene"}, files=files).json()
    assert res["task"] == "captioning"

def test_change_analysis_routing(synthetic_geotiff_bytes):
    files = [
        ("files", ("t1.tif", synthetic_geotiff_bytes, "image/tiff")),
        ("files", ("t2.tif", synthetic_geotiff_bytes, "image/tiff"))
    ]
    res = client.post("/analyze", data={"query": "What changed?"}, files=files).json()
    assert res["task"] == "change_analysis"

def test_optical_sar_routing(synthetic_geotiff_bytes):
    files = [
        ("files", ("opt.tif", synthetic_geotiff_bytes, "image/tiff")),
        ("files", ("sar.tif", synthetic_geotiff_bytes, "image/tiff"))
    ]
    res = client.post("/analyze", data={"query": "Analyze optical and SAR together"}, files=files).json()
    assert res["task"] == "optical_sar"

def test_ambiguous_two_image_routing(synthetic_geotiff_bytes):
    files = [
        ("files", ("1.tif", synthetic_geotiff_bytes, "image/tiff")),
        ("files", ("2.tif", synthetic_geotiff_bytes, "image/tiff"))
    ]
    res = client.post("/analyze", data={"query": "What is this?"}, files=files).json()
    assert res["status"] == "INVALID_INPUT"

def test_unpermitted_parameters(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Highlight", "parameters": '{"fake_hack": true}'}, files=files).json()
    assert res["status"] == "INVALID_INPUT"

def test_corrupted_tiff_rejected():
    files = [("files", ("bad.tif", b"corrupted_bytes", "image/tiff"))]
    res = client.post("/analyze", data={"query": "test"}, files=files).json()
    assert res["status"] == "INVALID_INPUT"
