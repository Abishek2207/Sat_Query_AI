import json
from typing import List
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalysisResponse
from .agent import app_graph
from .registry import MODEL_REGISTRY, get_tool_status
from .report import generate_pdf_report
from .history import log_request, get_history, get_stats

app = FastAPI(title="SatQuery AI API Gateway", version="1.0.0")

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
def health_check():
    tool_status = {task: get_tool_status(task) for task in MODEL_REGISTRY.keys()}
    return {
        "api_status": "AVAILABLE",
        "specialist_endpoints_configured": any(v["base_url"] != "" for v in MODEL_REGISTRY.values()),
        "registry_status": tool_status,
        "baseline_tools": ["change_map", "optical_sar"],
    }

@app.get("/evaluations")
def get_evaluations():
    import os
    from pathlib import Path
    benchmarks = ["vrsbench", "rsvqa", "cdvqa", "isro_sac"]
    results = []
    for b in benchmarks:
        path = Path(f"evaluation/{b}/results.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            results.append({
                "benchmark_name": b.upper(),
                "status": "NOT_CONFIGURED",
                "metrics": None,
                "expected_format": "Unknown",
                "dataset_path": "Unknown",
                "description": "Dataset unavailable / not connected."
            })
    return {"evaluations": results}

@app.get("/admin/stats")
def admin_stats():
    return get_stats()

@app.get("/admin/history")
def admin_history():
    return {"history": get_history()}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
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
