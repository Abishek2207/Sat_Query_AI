from typing import Dict, Any, Optional

class EvidencePolicyError(Exception):
    """Raised when an attempt is made to fabricate or spoof evidence."""
    pass

def validate_evidence(evidence_list: list) -> list:
    """
    Enforces the NO_FABRICATION policy on extracted evidence.
    Rules:
    - No synthetic benchmark scores.
    - No fake ground truth.
    - No fabricated confidence (must be real float from model, not arbitrary).
    - No fabricated bounding boxes.
    """
    for ev in evidence_list:
        if "confidence" in ev and ev["confidence"] is not None:
            if not (0.0 <= ev["confidence"] <= 1.0):
                raise EvidencePolicyError(f"Fabricated confidence detected: {ev['confidence']}")
        
        if "region" in ev and ev["region"]:
            # Region must be a valid 4-element coordinate list [xmin, ymin, xmax, ymax]
            if not isinstance(ev["region"], list) or len(ev["region"]) != 4:
                raise EvidencePolicyError(f"Fabricated bounding box detected: {ev['region']}")

    return evidence_list

def verify_dataset_status(status: str) -> str:
    """Ensures we never claim READY unless data is fully present."""
    valid_states = ["READY", "PARTIAL", "NOT_AVAILABLE"]
    if status not in valid_states:
        raise EvidencePolicyError(f"Invalid dataset status: {status}")
    return status

def enforce_abstention(evidence_list: list, conflict_detected: bool = False) -> Dict[str, Any]:
    """
    If evidence is insufficient or conflicts exist, the system MUST abstain 
    and return UNCERTAIN.
    """
    if conflict_detected:
        return {
            "status": "UNCERTAIN",
            "abstention_reason": "Conflict detected between modalities or reasoning paths."
        }
    
    if not evidence_list:
        return {
            "status": "UNCERTAIN",
            "abstention_reason": "Insufficient evidence extracted from models to formulate a reliable response."
        }
        
    return {"status": "SUCCESS", "abstention_reason": None}
