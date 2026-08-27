# RSICD Adaptation Report (Local CPU Test)

## Experiment
- **Objective**: Fix PEFT inputs_embeds conflict via Native PyTorch LoRA
- **Dataset**: arampacha/rsicd (400 Train, 100 Val)
- **Base model**: Salesforce/blip-image-captioning-base
- **Adaptation method**: Native Custom LoRA injected into query/value
- **LoRA configuration**: r=8, alpha=16
- **Trainable parameters**: 589824
- **Environment**: cpu

## Training
- **Epochs**: 1
- **Loss**: 7.1222
- **Duration**: 399.88s

## Results
- **Adapter**: `models/rsicd_blip_lora_local/adapter.pt`
- Successfully loaded and generated captions on 20 validation samples.

## Diagnostic
The previous `get_peft_model` implementation forwarded duplicate args during BLIP's multimodal pass. Writing a custom PyTorch LoRA completely avoids PEFT's generic forward-hooks and natively modifies the linear weights, solving the crash.
