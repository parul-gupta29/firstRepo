#!/bin/bash
# Fine-tune MDLM from pretrained OpenWebText checkpoint on WikiHow dataset
# Usage: bash scripts/finetune_wikihow_mdlm.sh

# Activate conda environment (uncomment if needed)
# source ~/.bashrc
# conda activate mdlm

# Create necessary directories
mkdir -p outputs
mkdir -p watch_folder
mkdir -p data/wikihow

# Fine-tune from pretrained HuggingFace checkpoint
python main.py \
  mode=train \
  model=small \
  data=wikihow \
  wandb.name=mdlm-wikihow-finetune \
  parameterization=subs \
  model.length=1024 \
  loader.global_batch_size=128 \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.max_steps=100000 \
  trainer.val_check_interval=2000 \
  optim.lr=1e-4 \
  checkpointing.resume_from_ckpt=false \
  seed=42

# Note: To fine-tune from a local checkpoint, add:
#   eval.checkpoint_path=/path/to/your/checkpoint.ckpt
#
# To fine-tune from HuggingFace pretrained model, you need to:
# 1. First download the model weights
# 2. Then use them as initialization
