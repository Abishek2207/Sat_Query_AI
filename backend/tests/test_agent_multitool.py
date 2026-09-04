import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_vqa(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "How many buildings?"}, files=files).json()
    assert res["task"] == "vqa"
    assert "vqa" in res["selected_tools"]

def test_captioning(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Describe the scene"}, files=files).json()
    assert res["task"] == "captioning"
    assert "captioning" in res["selected_tools"]

def test_grounding(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Highlight the water"}, files=files).json()
    assert res["task"] == "grounding"
    assert "grounding" in res["selected_tools"]

def test_land_cover(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "What is the dominant land cover?"}, files=files).json()
    assert res["task"] == "land_cover_classification"
    assert "land_cover_classification" in res["selected_tools"]

def test_temporal_change(synthetic_geotiff_bytes):
    files = [
        ("files", ("t1.tif", synthetic_geotiff_bytes, "image/tiff")),
        ("files", ("t2.tif", synthetic_geotiff_bytes, "image/tiff"))
    ]
    res = client.post("/analyze", data={"query": "What changed?"}, files=files).json()
    assert res["task"] == "change_analysis"
    assert "change_analysis" in res["selected_tools"]

def test_optical_sar(synthetic_geotiff_bytes):
    files = [
        ("files", ("opt.tif", synthetic_geotiff_bytes, "image/tiff")),
        ("files", ("sar.tif", synthetic_geotiff_bytes, "image/tiff"))
    ]
    res = client.post("/analyze", data={"query": "Analyze optical and SAR together"}, files=files).json()
    assert res["task"] == "optical_sar"
    assert "optical_sar" in res["selected_tools"]

def test_multi_tool(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Describe the image and highlight buildings"}, files=files).json()
    assert res["task"] == "multi_tool"
    assert "captioning" in res["selected_tools"]
    assert "grounding" in res["selected_tools"]
    assert res["parsed_intent"]["multi_tool"] is True

def test_missing_temporal_input(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "What changed?"}, files=files).json()
    assert res["status"] == "DATA_UNAVAILABLE"

def test_missing_sar_input(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Analyze SAR"}, files=files).json()
    assert res["status"] == "DATA_UNAVAILABLE"

def test_hallucination_trap(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    # We trigger a failure that generates a DATA_UNAVAILABLE or INSUFFICIENT_EVIDENCE
    # using unpermitted parameters to mock a failure.
    res = client.post("/analyze", data={"query": "Highlight", "parameters": '{"fake_hack": true}'}, files=files).json()
    assert res["status"] == "INVALID_INPUT"

def test_execution_trace(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Describe the scene"}, files=files).json()
    assert len(res["execution_trace"]) > 0
    assert any("QUERY_UNDERSTANDING" in t for t in res["execution_trace"])

def test_frontend_compatibility(synthetic_geotiff_bytes):
    files = [("files", ("test.tif", synthetic_geotiff_bytes, "image/tiff"))]
    res = client.post("/analyze", data={"query": "Describe the scene"}, files=files).json()
    # Check that required frontend fields exist
    assert "task" in res
    assert "status" in res
    assert "answer" in res
    assert "evidence" in res
    assert "conflict" in res
    assert "selected_tools" in res # Our new field
