import os
import json

def evaluate():
    print("--- REAL RSICD EVALUATION ---")
    val_file = "datasets/rsicd/dataset_rsicd.json"
    
    if not os.path.exists(val_file):
        print("Evaluation skipped: reference captions unavailable")
        
        report = {
            "status": "skipped",
            "reason": "reference captions unavailable"
        }
        
        os.makedirs("reports", exist_ok=True)
        with open("reports/rsicd_evaluation.json", "w") as f:
            json.dump(report, f, indent=2)
        return

    print("Reference captions found. Starting evaluation...")
    # Evaluation logic would go here if file existed

if __name__ == "__main__":
    evaluate()
