import httpx
from typing import List, Dict
from .registry import MODEL_REGISTRY, get_tool_status

async def call_specialist_model(task: str, query: str, files_data: List[Dict], params: dict) -> dict:
    permitted = MODEL_REGISTRY[task]["permitted_parameters"]
    safe_params = {k: v for k, v in params.items() if k in permitted}
    if len(safe_params) != len(params):
         return {"status": "INVALID_INPUT", "answer": "Unpermitted parameters rejected.", "task": task, "evidence": []}

    if get_tool_status(task) == "UNAVAILABLE" and MODEL_REGISTRY[task]["base_url"]:
        return {"status": "MODEL_UNAVAILABLE", "answer": "Remote endpoint unavailable.", "task": task, "evidence": []}

    base_url = MODEL_REGISTRY[task]["base_url"]

    # Fallback to local models if no endpoint is configured or if it's explicitly local
    if not base_url:
        from .local_specialists import run_local_vqa, run_local_captioning, run_local_grounding
        if task == "vqa":
            return run_local_vqa(query, files_data)
        elif task == "captioning":
            return run_local_captioning(files_data)
        elif task == "grounding":
            return run_local_grounding(query, files_data, params.get("confidence_threshold", 0.3))
        elif task == "change_analysis":
            from .change_map import compute_change_baseline
            return compute_change_baseline(files_data[0]["bytes"], files_data[1]["bytes"])
        elif task == "optical_sar":
            from .optical_sar import verify_optical_sar_pair
            return verify_optical_sar_pair(files_data[0]["bytes"], files_data[1]["bytes"])
        else:
            return {"status": "MODEL_UNAVAILABLE", "answer": "No local implementation for this task.", "task": task, "evidence": []}

    data = {"query": query, **safe_params}
    files_to_send = [("files", (f["filename"], f["bytes"], "application/octet-stream")) for f in files_data]

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(f"{base_url}/predict", data=data, files=files_to_send)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"status": "MODEL_UNAVAILABLE", "answer": "Model connection timeout.", "task": task, "evidence": []}
    except httpx.ConnectError:
        return {"status": "MODEL_UNAVAILABLE", "answer": "Model refused connection.", "task": task, "evidence": []}
    except Exception as e:
         return {"status": "MODEL_UNAVAILABLE", "answer": f"Unexpected error: {str(e)}", "task": task, "evidence": []}
