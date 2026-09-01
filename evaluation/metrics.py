"""
Evaluation Metrics for Remote Sensing Image Analysis.

Computes:
- Classification: Overall Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix
- Segmentation: IoU (Intersection-over-Union), Mean IoU, Dice Coefficient, Pixel Accuracy
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def calculate_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification performance metrics.
    
    Args:
        y_true: Ground truth class integer labels
        y_pred: Predicted class integer labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, per_class metrics, and confusion matrix.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    
    if len(y_t) == 0:
        return {"accuracy": 0.0, "error": "Empty predictions"}

    unique_labels = sorted(list(set(np.concatenate([y_t, y_p]))))
    num_classes = max(unique_labels) + 1 if unique_labels else 1
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    if HAS_SKLEARN:
        acc = float(accuracy_score(y_t, y_p))
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_t, y_p, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_t, y_p, average="weighted", zero_division=0
        )
        
        # Per class
        p_c, r_c, f_c, support = precision_recall_fscore_support(
            y_t, y_p, labels=list(range(len(class_names))), zero_division=0
        )
        
        per_class = {}
        for idx, c_name in enumerate(class_names):
            if idx < len(p_c):
                per_class[c_name] = {
                    "precision": round(float(p_c[idx]), 4),
                    "recall": round(float(r_c[idx]), 4),
                    "f1_score": round(float(f_c[idx]), 4),
                    "support": int(support[idx])
                }

        cm = confusion_matrix(y_t, y_p, labels=list(range(len(class_names))))
    else:
        # Pure NumPy fallback
        acc = float(np.mean(y_t == y_p))
        cm = np.zeros((len(class_names), len(class_names)), dtype=int)
        for t, p in zip(y_t, y_p):
            if t < len(class_names) and p < len(class_names):
                cm[t, p] += 1
                
        per_class = {}
        f1_list, prec_list, rec_list = [], [], []
        for i, c_name in enumerate(class_names):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            supp = int(np.sum(cm[i, :]))
            
            per_class[c_name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": supp
            }
            if supp > 0:
                f1_list.append(f1)
                prec_list.append(prec)
                rec_list.append(rec)
                
        prec_macro = float(np.mean(prec_list)) if prec_list else 0.0
        rec_macro = float(np.mean(rec_list)) if rec_list else 0.0
        f1_macro = float(np.mean(f1_list)) if f1_list else 0.0
        prec_weighted = prec_macro
        rec_weighted = rec_macro
        f1_weighted = f1_macro

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }


def calculate_segmentation_metrics(
    y_true_mask: np.ndarray,
    y_pred_mask: np.ndarray,
    num_classes: int = 2
) -> Dict[str, Any]:
    """
    Calculates pixel-level semantic segmentation metrics (IoU, mIoU, Dice).
    
    Args:
        y_true_mask: Ground truth 2D integer mask (H, W) or (N, H, W)
        y_pred_mask: Predicted 2D integer mask (H, W) or (N, H, W)
        num_classes: Number of distinct classes
        
    Returns:
        Dictionary with per-class IoU, Mean IoU (mIoU), Dice Score, and Pixel Accuracy
    """
    y_t = y_true_mask.flatten()
    y_p = y_pred_mask.flatten()

    pixel_acc = float(np.mean(y_t == y_p))
    iou_per_class = {}
    dice_per_class = {}
    ious = []

    for c in range(num_classes):
        true_c = (y_t == c)
        pred_c = (y_p == c)
        
        intersection = np.sum(true_c & pred_c)
        union = np.sum(true_c | pred_c)
        
        if union == 0:
            iou = 1.0 if np.sum(true_c) == 0 else 0.0
        else:
            iou = float(intersection) / float(union)
            
        dice = (2.0 * intersection) / (np.sum(true_c) + np.sum(pred_c) + 1e-7)
        
        iou_per_class[f"Class_{c}"] = round(iou, 4)
        dice_per_class[f"Class_{c}"] = round(float(dice), 4)
        ious.append(iou)

    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(list(dice_per_class.values())))

    return {
        "pixel_accuracy": round(pixel_acc, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "iou_per_class": iou_per_class,
        "dice_per_class": dice_per_class
    }


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Prints a clean CLI tabular summary of evaluation metrics."""
    print("\n" + "=" * 65)
    print("           REMOTE SENSING MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"  Overall Accuracy : {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro Precision  : {metrics.get('macro_precision', 0.0) * 100:.2f}%")
    print(f"  Macro Recall     : {metrics.get('macro_recall', 0.0) * 100:.2f}%")
    print(f"  Macro F1-Score   : {metrics.get('macro_f1', 0.0) * 100:.2f}%")
    print(f"  Weighted F1      : {metrics.get('weighted_f1', 0.0) * 100:.2f}%")
    print("-" * 65)
    print(f"{'Class Name':<24} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    
    per_class = metrics.get("per_class", {})
    for c_name, scores in per_class.items():
        p = scores.get('precision', 0.0) * 100
        r = scores.get('recall', 0.0) * 100
        f = scores.get('f1_score', 0.0) * 100
        s = scores.get('support', 0)
        print(f"{c_name:<24} | {p:>8.2f}% | {r:>8.2f}% | {f:>8.2f}% | {s:>7}")
        
    print("=" * 65 + "\n")
