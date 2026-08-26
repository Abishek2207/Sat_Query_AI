# RSICD Adaptation Report

1. **Dataset source:** `arampacha/rsicd`
2. **Dataset revision:** `main`
3. **Exact sample counts:** Train=400, Val=100
4. **Model identifier:** `Salesforce/blip-image-captioning-base`
5. **Model revision:** `main`
6. **LoRA configuration:** r=8, alpha=16, dropout=0.05, target_modules=["query", "value", "q_proj", "v_proj"]
7. **Hardware/device:** CPU
8. **Training duration:** FAILED (aborted during initial forward pass)
9. **Final training loss:** N/A
10. **Validation result:** Baseline executed on 20 samples.
11. **Baseline vs adapted result:** Adaptation failed, no adapted predictions generated.
12. **Exact checkpoint location:** N/A
13. **Any limitations:** 
   **SOFTWARE LIMITATION ENCOUNTERED.** The PEFT library conflicts with the Hugging Face `BlipForConditionalGeneration` architecture when used within the standard `Trainer` pipeline. Wrapping the model with LoRA causes the `BlipEncoder` to receive duplicate arguments during the forward pass:
   `TypeError: BlipEncoder(...) got multiple values for keyword argument 'inputs_embeds'`
   Per strict SIH directives, training was immediately halted. No fake checkpoints or fabricated metrics were generated.
