#!/bin/bash
# Full fine-tune pretrained MDLM (small) from HuggingFace on WikiHow dataset
# Unlike LoRA fine-tuning, this script updates ALL model parameters.
# Usage: bash scripts/full_finetune_wikihow_mdlm.sh
#
# Requirements:
#   - Multiple GPUs recommended (all parameters are trained)
#   - More memory than LoRA (~2-3x); bf16 precision helps
#   - Longer training schedule with lower learning rate

# Activate conda environment (uncomment if needed)
# source ~/.bashrc
# conda activate mdlm

# Create necessary directories
mkdir -p outputs
mkdir -p watch_folder
mkdir -p data/wikihow

# Full fine-tune from pretrained HuggingFace checkpoint (kuleshov-group/mdlm-owt)
# Key differences from LoRA fine-tuning:
#   - lora.enabled=false (no LoRA adapters; all weights are trainable)
#   - Lower learning rate (3e-5 vs 2e-4) to avoid catastrophic forgetting
#   - Larger batch size for stable gradient estimates
#   - More training steps to fully adapt all parameters
#   - EMA enabled (training.ema) for smoother convergence
#   - Cosine decay LR schedule with warmup
python main.py \
  mode=train \
  model=small \
  data=wikihow \
  wandb.name=mdlm-wikihow-small-full-ft \
  parameterization=subs \
  model.length=1024 \
  pretrained_model_name=kuleshov-group/mdlm-owt \
  lora.enabled=false \
  loader.global_batch_size=512 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.max_steps=50000 \
  trainer.val_check_interval=5000 \
  trainer.devices=4 \
  trainer.gradient_clip_val=1.0 \
  trainer.precision=bf16 \
  training.ema=0.9999 \
  optim.lr=5e-6 \
  optim.weight_decay=0.01 \
  lr_scheduler=cosine_decay_warmup \
  checkpointing.resume_from_ckpt=false \
  seed=42