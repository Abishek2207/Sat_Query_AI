from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from .validator import validate_image_bytes
from .adapters import call_specialist_model
from .schemas import ValidationResult, EvidenceItem
from .evidence_policy import validate_evidence, enforce_abstention, EvidencePolicyError

class AgentState(TypedDict):
    query: str
    files: List[Dict[str, Any]]
    parameters: dict
    benchmark_mode: bool
    validation_results: List[ValidationResult]
    parsed_intent: dict
    selected_task: str
    api_response: dict
    trace: List[str]
    app_state: Any

def validate_node(state: AgentState):
    trace = state.get("trace", [])
    val_results = []
    
    # Check single image validations
    for f in state["files"]:
        res = validate_image_bytes(f["filename"], f["bytes"], benchmark_mode=state.get("benchmark_mode", False))
        val_results.append(res)
        if not res.valid:
            trace.append(f"VALIDATION FAILED: {f['filename']} -> {res.reason}")
            return {"api_response": {"status": "INVALID_INPUT", "answer": "VALIDATION_FAILED: " + res.reason}, "validation_results": val_results, "trace": trace}
        trace.append(f"VALIDATED: {f['filename']} (Modality: {res.modality}, Bands: {res.bands})")
        
    # Check two-image pair compatibility
    if len(val_results) == 2:
        from .validator import validate_image_pair
        pair_res = validate_image_pair(val_results[0], val_results[1])
        if not pair_res.valid:
            trace.append(f"PAIR VALIDATION FAILED: {pair_res.reason}")
            return {"api_response": {"status": "ABSTAINED", "answer": "ABSTAINED: " + pair_res.reason, "abstention_reason": pair_res.reason}, "validation_results": val_results, "trace": trace}
            
    return {"validation_results": val_results, "trace": trace}

def understand_node(state: AgentState):
    if state.get("api_response", {}).get("status") == "INVALID_INPUT": return state
    query = state["query"].lower()
    
    intent = {"requires_spatial": False, "requires_temporal": False, "requires_crossmodal": False}
    if any(kw in query for kw in ["where", "locate", "highlight", "find"]):
        intent["requires_spatial"] = True
    if any(kw in query for kw in ["change", "difference"]):
        intent["requires_temporal"] = True
    if any(kw in query for kw in ["sar", "radar"]) and any(kw in query for kw in ["optical"]):
        intent["requires_crossmodal"] = True
        
    state["trace"].append(f"UNDERSTANDING: Parsed intent -> {intent}")
    return {"parsed_intent": intent, "trace": state["trace"]}

def route_node(state: AgentState):
    if state.get("api_response", {}).get("status") == "INVALID_INPUT": return state
    
    query = state["query"].lower()
    img_count = len(state["files"])
    intent = state.get("parsed_intent", {})
    trace = state["trace"]
    val = state["validation_results"]

    task = "UNKNOWN"
    if img_count == 2:
        if intent["requires_crossmodal"] or (val[0].modality != val[1].modality and val[0].modality != "unknown" and val[1].modality != "unknown"):
            task = "optical_sar"
        elif intent["requires_temporal"] or any(kw in query for kw in ["change", "difference"]):
            task = "change_analysis"
    elif img_count == 1:
        lc_keywords = [
            "classify", "classification", "land cover", "land-cover", 
            "land surface", "land type", "terrain type", "area type", 
            "geographic land", "what type of area", "what kind of land", 
            "what kind of area", "identify the land"
        ]
        if any(kw in query for kw in lc_keywords):
            task = "land_cover_classification"
        elif intent["requires_spatial"]:
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
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "ABSTAINED"]: return state
    trace = state["trace"]
    res = await call_specialist_model(state["selected_task"], state["query"], state["files"], state["parameters"], state.get("app_state"))
    trace.append(f"EXECUTION: Received status {res.get('status')}")
    return {"api_response": res, "trace": trace}

def aggregate_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "MODEL_UNAVAILABLE"]: return state
    res = state.get("api_response", {})
    trace = state["trace"]
    
    if "evidence" not in res:
        res["evidence"] = []
        
    trace.append(f"AGGREGATION: Aggregated {len(res['evidence'])} evidence items.")
    return {"api_response": res, "trace": trace}

def verify_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "MODEL_UNAVAILABLE"]: return state
    res = state.get("api_response", {})
    trace = state["trace"]
    task = state.get("selected_task")
    
    # Enforce NO_FABRICATION evidence policy
    try:
        res["evidence"] = validate_evidence(res.get("evidence", []))
    except EvidencePolicyError as e:
        trace.append(f"VERIFICATION FAILED: {str(e)}")
        res["status"] = "DATA_UNAVAILABLE"
        return {"api_response": res, "trace": trace}
        
    if len(res.get("evidence", [])) == 0:
        trace.append("VERIFICATION: No evidence items returned.")
        res["status"] = "DATA_UNAVAILABLE"
        return {"api_response": res, "trace": trace}
        
    if task == "grounding":
        has_region = any(ev.get("region") is not None for ev in res["evidence"] if isinstance(ev, dict))
        if not has_region:
            trace.append("VERIFICATION: Grounding requested but no spatial region found in evidence.")
            res["status"] = "DATA_UNAVAILABLE"
            
    return {"api_response": res, "trace": trace}

def conflict_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "MODEL_UNAVAILABLE", "DATA_UNAVAILABLE"]: return state
    res = state.get("api_response", {})
    trace = state["trace"]
    
    if res.get("conflict") is True:
        trace.append("CONFLICT DETECTION: Conflict explicitly flagged by specialist.")
        # Uncertainty will be handled in abstain_node
        if res.get("confidence") is not None:
            res["confidence"] = max(0.0, res["confidence"] - 0.4)
            
    return {"api_response": res, "trace": trace}

def abstain_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "MODEL_UNAVAILABLE"]: return state
    res = state.get("api_response", {})
    trace = state["trace"]
    
    if res.get("status") == "DATA_UNAVAILABLE":
        trace.append("ABSTENTION: Abstaining due to missing data/evidence.")
        res["abstention_reason"] = "Required evidence could not be generated."
        res["answer"] = "DATA_UNAVAILABLE: " + res["abstention_reason"]
    else:
        # Check abstention via policy
        policy_res = enforce_abstention(res.get("evidence", []), res.get("conflict", False))
        if policy_res["status"] == "UNCERTAIN":
            res["status"] = "UNCERTAIN"
            res["abstention_reason"] = policy_res["abstention_reason"]
            trace.append(f"ABSTENTION: {policy_res['abstention_reason']}")
            if not res.get("conflict", False):
                res["answer"] = "UNCERTAIN: " + policy_res["abstention_reason"]

    return {"api_response": res, "trace": trace}

workflow = StateGraph(AgentState)
workflow.add_node("validate", validate_node)
workflow.add_node("understand", understand_node)
workflow.add_node("route", route_node)
workflow.add_node("execute", execute_node)
workflow.add_node("aggregate", aggregate_node)
workflow.add_node("verify", verify_node)
workflow.add_node("conflict", conflict_node)
workflow.add_node("abstain", abstain_node)

workflow.set_entry_point("validate")
workflow.add_edge("validate", "understand")
workflow.add_edge("understand", "route")
workflow.add_edge("route", "execute")
workflow.add_edge("execute", "aggregate")
workflow.add_edge("aggregate", "verify")
workflow.add_edge("verify", "conflict")
workflow.add_edge("conflict", "abstain")
workflow.add_edge("abstain", END)

app_graph = workflow.compile()
