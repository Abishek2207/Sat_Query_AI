import time
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
    selected_tools: List[str]
    tool_results: List[dict]
    selected_task: str
    api_response: dict
    trace: List[str]
    app_state: Any
    warnings: List[str]

def validate_node(state: AgentState):
    t0 = time.time()
    t0_validate_node = time.time()
    trace = state.get("trace", [])
    val_results = []
    
    for f in state["files"]:
        res = validate_image_bytes(f["filename"], f["bytes"], benchmark_mode=state.get("benchmark_mode", False))
        val_results.append(res)
        if not res.valid:
            trace.append(f"INPUT_VALIDATION FAILED: {f['filename']} -> {res.reason}")
            return {"api_response": {"status": "INVALID_INPUT", "answer": "VALIDATION_FAILED: " + res.reason}, "validation_results": val_results, "trace": trace}
        trace.append(f"INPUT_VALIDATION: {f['filename']} (Modality: {res.modality}, Bands: {res.bands})")
        
    if len(val_results) == 2:
        from .validator import validate_image_pair
        pair_res = validate_image_pair(val_results[0], val_results[1])
        if not pair_res.valid:
            trace.append(f"INPUT_VALIDATION FAILED (PAIR): {pair_res.reason}")
            return {"api_response": {"status": "DATA_UNAVAILABLE", "answer": "DATA_UNAVAILABLE: " + pair_res.reason, "abstention_reason": pair_res.reason}, "validation_results": val_results, "trace": trace}
            
    print(f"[PERF] file validation: {time.time() - t0:.2f}s")
    return {"validation_results": val_results, "trace": trace}

def parse_query_node(state: AgentState):
    t0 = time.time()
    t0_parse_query_node = time.time()
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "DATA_UNAVAILABLE"]: return state
    query = state["query"].lower()
    
    intent = {
        "vqa": False,
        "captioning": False,
        "grounding": False,
        "land_cover": False,
        "change_analysis": False,
        "optical_sar": False,
        "multi_tool": False
    }

    lc_keywords = [
        "classify", "classification", "land cover", "land-cover", 
        "land surface", "land type", "terrain type", "area type", 
        "geographic land", "what type of area", "what kind of land", 
        "what kind of area", "identify the land"
    ]
    if any(kw in query for kw in lc_keywords):
        intent["land_cover"] = True

    if any(kw in query for kw in ["where", "locate", "highlight", "find"]):
        intent["grounding"] = True

    if any(kw in query for kw in ["describe", "scene", "caption"]):
        intent["captioning"] = True

    if any(kw in query for kw in ["change", "difference"]):
        intent["change_analysis"] = True

    if any(kw in query for kw in ["sar", "radar"]):
        intent["optical_sar"] = True
        
    if not any(intent.values()):
        intent["vqa"] = True
        
    if sum([1 for k, v in intent.items() if v]) > 1:
        intent["multi_tool"] = True
        
    trace = state.get("trace", [])
    trace.append(f"QUERY_UNDERSTANDING: Parsed intents -> {intent}")
    print(f"[PERF] query parsing: {time.time() - t0:.2f}s")
    return {"parsed_intent": intent, "trace": trace}

def plan_tools_node(state: AgentState):
    t0 = time.time()
    t0_plan_tools_node = time.time()
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "DATA_UNAVAILABLE"]: return state
    
    intent = state.get("parsed_intent", {})
    trace = state["trace"]
    img_count = len(state["files"])
    val = state["validation_results"]

    tools = []
    warnings = state.get("warnings", [])

    if img_count == 2:
        if intent.get("change_analysis"):
            tools.append("change_analysis")
        elif intent.get("optical_sar") or (val[0].modality != val[1].modality and val[0].modality != "unknown" and val[1].modality != "unknown"):
            tools.append("optical_sar")
        # Do not force change_analysis if intent is completely unknown to preserve ambiguity
    elif img_count == 1:
        if intent.get("change_analysis") or intent.get("optical_sar"):
            trace.append("TOOL_SELECTION FAILED: Requested multi-image capability but provided only 1 image.")
            return {"api_response": {"status": "DATA_UNAVAILABLE", "answer": "DATA_UNAVAILABLE: This operation requires two compatible images."}, "trace": trace}
            
        if intent.get("captioning"):
            tools.append("captioning")
        if intent.get("grounding"):
            tools.append("grounding")
        if intent.get("land_cover"):
            tools.append("land_cover_classification")
        if intent.get("vqa") and not tools:
            tools.append("vqa")
            
    if not tools:
        trace.append("TOOL_SELECTION FAILED: Ambiguous or unsupported input configuration.")
        return {"api_response": {"status": "INVALID_INPUT", "answer": "Ambiguous configuration."}, "trace": trace}

    print(f"[PERF] specialist selection: {time.time() - t0:.2f}s")
    return {"selected_tools": tools, "trace": trace, "warnings": warnings}

async def execute_tools_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "DATA_UNAVAILABLE"]: return state
    trace = state["trace"]
    tools = state.get("selected_tools", [])
    
    tool_results = []
    for tool in tools:
        res = await call_specialist_model(tool, state["query"], state["files"], state["parameters"], state.get("app_state"))
        res["tool"] = tool
        tool_results.append(res)
        trace.append(f"EXECUTION: Tool '{tool}' returned status {res.get('status')}")
        
    return {"tool_results": tool_results, "trace": trace}

def synthesize_results_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "DATA_UNAVAILABLE"]: return state
    trace = state["trace"]
    results = state.get("tool_results", [])
    tools = state.get("selected_tools", [])
    
    combined_evidence = []
    combined_answers = []
    any_failed = False
    all_failed = True
    combined_visual = None
    has_conflict = False
    
    confidences = []
    
    for r in results:
        if r.get("status") == "SUCCESS":
            all_failed = False
            combined_answers.append(f"[{r['tool'].upper()}]: {r.get('answer', '')}")
            if r.get("evidence"):
                combined_evidence.extend(r["evidence"])
            if r.get("visual_output") and not combined_visual:
                combined_visual = r["visual_output"]
            if r.get("confidence") is not None:
                confidences.append(r["confidence"])
            if r.get("conflict"):
                has_conflict = True
        else:
            any_failed = True
            combined_answers.append(f"[{r['tool'].upper()} FAILED]: {r.get('answer', 'Unknown error')}")
            
    if all_failed:
        # Check if any tool returned INVALID_INPUT specifically
        if any(r.get("status") == "INVALID_INPUT" for r in results):
            status = "INVALID_INPUT"
        else:
            status = "MODEL_UNAVAILABLE"
        trace.append(f"SYNTHESIS: All tools failed with status {status}.")
    elif any_failed:
        status = "PARTIALLY_VERIFIED"
        trace.append("SYNTHESIS: Partial success. Some tools failed.")
    else:
        status = "SUCCESS"
        trace.append("SYNTHESIS: All tools succeeded.")
        
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    
    api_response = {
        "status": status,
        "answer": "\n\n".join(combined_answers),
        "evidence": combined_evidence,
        "visual_output": combined_visual,
        "confidence": avg_conf,
        "conflict": has_conflict
    }
    
    selected_task = tools[0] if len(tools) == 1 else "multi_tool"
    
    return {"api_response": api_response, "trace": trace, "selected_task": selected_task}

def verify_evidence_node(state: AgentState):
    t0 = time.time()
    t0_verify_evidence_node = time.time()
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "MODEL_UNAVAILABLE", "DATA_UNAVAILABLE"]: return state
    res = state.get("api_response", {})
    trace = state["trace"]
    task = state.get("selected_task")
    tools = state.get("selected_tools", [])
    
    try:
        res["evidence"] = validate_evidence(res.get("evidence", []))
    except EvidencePolicyError as e:
        trace.append(f"EVIDENCE_CHECK FAILED: {str(e)}")
        res["status"] = "DATA_UNAVAILABLE"
        print(f"[PERF] evidence verification: {time.time() - t0:.2f}s")
    return {"api_response": res, "trace": trace}
        
    if len(res.get("evidence", [])) == 0 and "change_analysis" not in tools and "optical_sar" not in tools:
        trace.append("EVIDENCE_CHECK: No evidence items returned for visual task.")
        res["status"] = "DATA_UNAVAILABLE"
        print(f"[PERF] evidence verification: {time.time() - t0:.2f}s")
    return {"api_response": res, "trace": trace}
        
    if "grounding" in tools:
        has_region = any(ev.get("region") is not None for ev in res["evidence"] if isinstance(ev, dict))
        if not has_region:
            trace.append("EVIDENCE_CHECK: Grounding requested but no spatial region found in evidence.")
            res["status"] = "DATA_UNAVAILABLE"
            
    trace.append("EVIDENCE_CHECK: SUCCESS")
    print(f"[PERF] evidence verification: {time.time() - t0:.2f}s")
    return {"api_response": res, "trace": trace}

def confidence_node(state: AgentState):
    if state.get("api_response", {}).get("status") in ["INVALID_INPUT", "MODEL_UNAVAILABLE", "DATA_UNAVAILABLE"]: return state
    res = state.get("api_response", {})
    trace = state["trace"]
    
    if res.get("conflict") is True:
        trace.append("CONFIDENCE_CALCULATION: Conflict explicitly flagged by specialist. Reducing confidence.")
        if res.get("confidence") is not None:
            res["confidence"] = max(0.0, res["confidence"] - 0.4)
            
    trace.append("CONFIDENCE_CALCULATION: SUCCESS")
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
        policy_res = enforce_abstention(res.get("evidence", []), res.get("conflict", False))
        
        if policy_res["status"] == "INSUFFICIENT_EVIDENCE":
            res["status"] = policy_res["status"]
            res["abstention_reason"] = policy_res["abstention_reason"]
            trace.append(f"ABSTENTION: {policy_res['abstention_reason']}")
            res["answer"] = "INSUFFICIENT EVIDENCE: " + policy_res["abstention_reason"]
            
        elif policy_res["status"] == "PARTIALLY_VERIFIED":
            if res["status"] == "SUCCESS":
                res["status"] = policy_res["status"]
            res["abstention_reason"] = policy_res["abstention_reason"]
            trace.append(f"ABSTENTION (PARTIAL): {policy_res['abstention_reason']}")

    return {"api_response": res, "trace": trace}

workflow = StateGraph(AgentState)
workflow.add_node("validate_inputs", validate_node)
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("plan_tools", plan_tools_node)
workflow.add_node("execute_tools", execute_tools_node)
workflow.add_node("synthesize_results", synthesize_results_node)
workflow.add_node("verify_evidence", verify_evidence_node)
workflow.add_node("calculate_confidence", confidence_node)
workflow.add_node("abstain", abstain_node)

workflow.set_entry_point("validate_inputs")
workflow.add_edge("validate_inputs", "parse_query")
workflow.add_edge("parse_query", "plan_tools")
workflow.add_edge("plan_tools", "execute_tools")
workflow.add_edge("execute_tools", "synthesize_results")
workflow.add_edge("synthesize_results", "verify_evidence")
workflow.add_edge("verify_evidence", "calculate_confidence")
workflow.add_edge("calculate_confidence", "abstain")
workflow.add_edge("abstain", END)

app_graph = workflow.compile()
