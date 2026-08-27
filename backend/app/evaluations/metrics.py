def calculate_iou(boxA, boxB):
    # box: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def exact_match_accuracy(predictions, references):
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if str(p).strip().lower() == str(r).strip().lower())
    return correct / len(predictions)

def simple_caption_match(pred, ref):
    # A basic BLEU-1 proxy for when NLTK/COCO-eval aren't available locally
    pred_words = set(str(pred).lower().split())
    ref_words = set(str(ref).lower().split())
    if not ref_words: return 0.0
    return len(pred_words.intersection(ref_words)) / len(ref_words)
