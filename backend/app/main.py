import json
import os
import torch
import gc
from contextlib import asynccontextmanager
from typing import List
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalysisResponse
from .agent import app_graph
from .model_registry import MODEL_REGISTRY, get_tool_status
from .report import generate_pdf_report
from .history import log_request, get_history, get_stats

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing models...")
    app.state.caption_model = None
    app.state.caption_processor = None
    app.state.caption_device = None
    app.state.adapter_loaded = False
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    adapter_path = os.path.join(project_root, "models", "rsicd_blip_lora", "adapter.pt")
    
    try:
        if os.path.exists(adapter_path):
            from src.inference.caption import RSICDCaptioner
            captioner = RSICDCaptioner(adapter_path=adapter_path)
            app.state.caption_model = captioner.model
            app.state.caption_processor = captioner.processor
            app.state.caption_device = captioner.device
            app.state.adapter_loaded = True
            print("[Startup] RSICDCaptioner loaded successfully WITH adapter.")
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            model.eval()
            app.state.caption_model = model
            app.state.caption_processor = processor
            app.state.caption_device = device
            app.state.adapter_loaded = False
            print("[Startup] Base model loaded successfully WITHOUT adapter.")
    except Exception as e:
        print(f"[Startup] Error initializing captioning model: {e}")
        
    yield
    print("[Shutdown] Cleaning up...")
    app.state.caption_model = None
    app.state.caption_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

app = FastAPI(title="SatQuery AI API Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://satquery-ai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"name": "SatQuery AI", "status": "AVAILABLE", "version": "1.0.0"}

@app.get("/health")
def health_check(request: Request):
    tool_status = {task: get_tool_status(task) for task in MODEL_REGISTRY.keys()}
    
    # Determine base model and adapter status
    base_model_loaded = getattr(request.app.state, "caption_model", None) is not None
    adapter_loaded = getattr(request.app.state, "adapter_loaded", False)
    
    # Determine overall status
    if not base_model_loaded:
        status = "offline"
    elif not adapter_loaded:
        status = "degraded"
    else:
        status = "ok"
        
    capabilities = {
        "captioning": tool_status.get("captioning") == "READY",
        "vqa": tool_status.get("vqa") == "READY",
        "grounding": tool_status.get("grounding") == "READY",
        "change_analysis": tool_status.get("change_analysis") == "READY",
        "optical_sar": tool_status.get("optical_sar") == "READY",
    }
    
    return {
        "status": status,
        "device": getattr(request.app.state, "caption_device", "unknown"),
        "base_model_loaded": base_model_loaded,
        "adapter_loaded": adapter_loaded,
        "capabilities": capabilities,
        "registry_status": tool_status
    }

@app.get("/models")
def get_models():
    from .model_registry import get_model_registry_list
    return {"models": get_model_registry_list()}

@app.get("/evaluations")
def get_evaluations():
    from .dataset_registry import get_dataset_registry
    registry = get_dataset_registry()
    results = []
    
    for ds in registry:
        results.append({
            "benchmark_name": ds["name"],
            "dataset_id": ds["dataset_id"],
            "status": ds["status"],
            "samples": ds["samples"],
            "metrics": None,
            "expected_format": "JSON/Images",
            "dataset_path": f"datasets/{ds['dataset_id']}",
            "description": f"Tasks: {', '.join(ds['task_types'])}"
        })
    return {"evaluations": results}

@app.post("/evaluations/{dataset_id}/run")
async def trigger_evaluation(dataset_id: str, limit: int = 5):
    from .evaluations.runner import run_benchmark_evaluation
    return await run_benchmark_evaluation(dataset_id, limit)

@app.get("/admin/stats")
def admin_stats():
    return get_stats()

@app.get("/admin/history")
def admin_history():
    return {"history": get_history()}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    query: str = Form(...),
    parameters: str = Form("{}"),
    benchmark_mode: bool = Form(False),
    files: List[UploadFile] = File(...),
):
    try:
        params_dict = json.loads(parameters)
    except Exception:
        params_dict = {}

    file_data = [{"filename": f.filename, "bytes": await f.read()} for f in files]

    state = {
        "query": query,
        "files": file_data,
        "parameters": params_dict,
        "benchmark_mode": benchmark_mode,
        "trace": [],
        "app_state": request.app.state,
    }

    try:
        result = await app_graph.ainvoke(state)
        api_res = result.get("api_response", {})

        response_obj = AnalysisResponse(
            task=result.get("selected_task", "unknown"),
            answer=api_res.get("answer", "ERROR"),
            status=api_res.get("status", "ERROR"),
            evidence=api_res.get("evidence", []),
            visual_output=api_res.get("visual_output"),
            confidence=api_res.get("confidence"),
            execution_trace=result.get("trace", []),
            provenance=api_res.get("provenance"),
            validation=result.get("validation_results"),
            conflict=api_res.get("conflict", False),
            abstention_reason=api_res.get("abstention_reason")
        )
        
        # Log to history DB
        log_request(query, response_obj.model_dump())

        return response_obj
    except Exception as e:
        return AnalysisResponse(
            task="unknown",
            status="ERROR",
            answer="An internal server error occurred during analysis.",
            evidence=[],
            execution_trace=["ERROR: " + str(e)]
        )

@app.post("/report")
async def download_report(
    request: Request,
    query: str = Form(...),
    parameters: str = Form("{}"),
    benchmark_mode: bool = Form(False),
    files: List[UploadFile] = File(...),
):
    try:
        params_dict = json.loads(parameters)
    except Exception:
        params_dict = {}

    file_data = [{"filename": f.filename, "bytes": await f.read()} for f in files]

    state = {
        "query": query,
        "files": file_data,
        "parameters": params_dict,
        "benchmark_mode": benchmark_mode,
        "trace": [],
        "app_state": request.app.state,
    }

    try:
        result = await app_graph.ainvoke(state)
        api_res = result.get("api_response", {})

        response_obj = AnalysisResponse(
            task=result.get("selected_task", "unknown"),
            answer=api_res.get("answer", "ERROR"),
            status=api_res.get("status", "ERROR"),
            evidence=api_res.get("evidence", []),
            visual_output=None,
            confidence=api_res.get("confidence"),
            execution_trace=result.get("trace", []),
            provenance=api_res.get("provenance"),
            validation=result.get("validation_results"),
            conflict=api_res.get("conflict", False),
            abstention_reason=api_res.get("abstention_reason")
        )

        report_bytes = generate_pdf_report(response_obj, query)
        return Response(
            content=report_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=SatQuery_Report.pdf"},
        )
    except Exception as e:
        return Response(content=f"Report generation failed: {str(e)}", status_code=500)
