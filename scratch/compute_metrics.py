import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    from rouge_score import rouge_scorer
    has_rouge = True
except ImportError:
    has_rouge = False

def compute_metrics(predictions_file):
    with open(predictions_file) as f:
        data = [json.loads(l) for l in f]
        
    bleu_scores = []
    rouge_l_scores = []
    
    if has_rouge:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    smoothie = SmoothingFunction().method4
        
    for item in data:
        ref = item["reference_caption"]
        pred = item.get("adapted_prediction") or item.get("baseline_prediction")
        
        # BLEU
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        pred_tokens = nltk.word_tokenize(pred.lower())
        b_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(b_score)
        
        # ROUGE
        if has_rouge:
            r_score = scorer.score(ref, pred)['rougeL'].fmeasure
            rouge_l_scores.append(r_score)
            
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores) if has_rouge else 0.0
    
    return avg_bleu, avg_rouge

def main():
    print("Computing Baseline Metrics...")
    base_bleu, base_rouge = compute_metrics("datasets/rsicd/baseline_predictions.jsonl")
    
    print("Computing Adapted Metrics...")
    adapt_bleu, adapt_rouge = compute_metrics("datasets/rsicd/adapted_predictions_local.jsonl")
    
    print(f"Baseline BLEU: {base_bleu:.6f}")
    print(f"Baseline ROUGE-L: {base_rouge:.6f}")
    print(f"Adapted BLEU: {adapt_bleu:.6f}")
    print(f"Adapted ROUGE-L: {adapt_rouge:.6f}")
    
    # Update results json
    with open("adaptation_results.json") as f:
        res = json.load(f)
        
    res["baseline_metrics"] = {"bleu": base_bleu, "rouge_l": base_rouge}
    res["adapted_metrics"] = {"bleu": adapt_bleu, "rouge_l": adapt_rouge}
    
    with open("adaptation_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    main()
