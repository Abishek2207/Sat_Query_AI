from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from .validator import validate_image_bytes
from .adapters import call_specialist_model
from .schemas import ValidationResult

class AgentState(TypedDict):
    query: str
    files: List[Dict[str, Any]]
    parameters: dict
    benchmark_mode: bool
    validation_results: List[ValidationResult]
    selected_task: str
    api_response: dict
    trace: List[str]

def validate_node(state: AgentState):
    trace = state.get("trace", [])
    val_results = []
    
    for f in state["files"]:
        res = validate_image_bytes(f["filename"], f["bytes"], benchmark_mode=state.get("benchmark_mode", False))
        val_results.append(res)
        if not res.valid:
            trace.append(f"VALIDATION FAILED: {f['filename']} -> {res.reason}")
            return {"api_response": {"status": "INVALID_INPUT", "answer": "INVALID_INPUT"}, "validation_results": val_results, "trace": trace}
        trace.append(f"VALIDATED: {f['filename']} (Format: {res.file_format})")
        
    return {"validation_results": val_results, "trace": trace}

def route_node(state: AgentState):
    if state.get("api_response", {}).get("status") == "INVALID_INPUT": return state
    
    query = state["query"].lower()
    img_count = len(state["files"])
    trace = state["trace"]

    task = "UNKNOWN"
    if img_count == 2:
        if any(kw in query for kw in ["sar", "radar", "optical"]):
            task = "optical_sar"
        elif any(kw in query for kw in ["change", "difference"]):
            task = "change_analysis"
    elif img_count == 1:
        if any(kw in query for kw in ["classify", "land cover", "class", "type", "euro"]):
            task = "land_cover_classification"
        elif any(kw in query for kw in ["highlight", "locate", "where", "find"]):
            task = "grounding"
        elif any(kw in query for kw in ["describe", "scene", "caption"]):
            task = "captioning"
        else:
            task = "vqa"
            
    if task == "UNKNOWN":
        trace.append("ROUTER FAILED: Ambiguous or unsupported input configuration.")
        return {"api_response": {"status": "INVALID_INPUT", "answer": "Ambiguous configuration."}, "trace": trace}

    trace.append(f"ROUTER: Selected task '{task}'.")
    return {"selected_task": task, "trace": trace}

async def execute_node(state: AgentState):
    if state.get("api_response", {}).get("status") == "INVALID_INPUT": return state
    
    trace = state["trace"]
    res = await call_specialist_model(state["selected_task"], state["query"], state["files"], state["parameters"])
    trace.append(f"EXECUTION: Received status {res.get('status')}")
    
    return {"api_response": res, "trace": trace}

def evidence_node(state: AgentState):
    res = state.get("api_response", {})
    trace = state["trace"]
    
    if res.get("status") in ["MODEL_UNAVAILABLE", "INVALID_INPUT"]: return state
    task = state.get("selected_task")
    
    prov = res.get("provenance")
    if not prov:
         trace.append("EVIDENCE VALIDATOR: Missing provenance.")
         return {"api_response": {"status": "EVIDENCE_UNAVAILABLE", "answer": "EVIDENCE_UNAVAILABLE", "evidence": []}, "trace": trace}
    
    if prov.get("remote_sensing_adapted") and not prov.get("adaptation_dataset"):
         trace.append("EVIDENCE VALIDATOR: Rejected fake remote_sensing_adapted claim.")
         res["provenance"]["remote_sensing_adapted"] = False

    has_visual = bool(res.get("visual_output"))
    has_text = bool(res.get("evidence") and len(res["evidence"]) > 0)
    
    if task not in ["vqa", "captioning"]:
        if not has_visual and not has_text:
            trace.append(f"EVIDENCE VALIDATOR: Task '{task}' lacks required spatial/visual evidence.")
            res["status"] = "EVIDENCE_UNAVAILABLE"
            res["answer"] = "EVIDENCE_UNAVAILABLE"
            res["confidence"] = None

    return {"api_response": res, "trace": trace}

workflow = StateGraph(AgentState)
workflow.add_node("validate", validate_node)
workflow.add_node("route", route_node)
workflow.add_node("execute", execute_node)
workflow.add_node("evidence", evidence_node)
workflow.set_entry_point("validate")
workflow.add_edge("validate", "route")
workflow.add_edge("route", "execute")
workflow.add_edge("execute", "evidence")
workflow.add_edge("evidence", END)
app_graph = workflow.compile()
