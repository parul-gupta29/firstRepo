#!/bin/bash
# Fine-tune pretrained MDLM (small) from HuggingFace on WikiHow dataset
# Usage: bash scripts/finetune_wikihow_mdlm.sh

# Activate conda environment (uncomment if needed)
# source ~/.bashrc
# conda activate mdlm

# Create necessary directories
mkdir -p outputs
mkdir -p watch_folder
mkdir -p data/wikihow

# Fine-tune from pretrained HuggingFace checkpoint (kuleshov-group/mdlm-owt)
python main.py \
  mode=train \
  model=small \
  data=wikihow \
  wandb.name=mdlm-wikihow-small-finetune \
  parameterization=subs \
  model.length=1024 \
  pretrained_model_name=kuleshov-group/mdlm-owt \
  loader.global_batch_size=128 \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.max_steps=20000 \
  trainer.val_check_interval=2000 \
  trainer.devices=2 \
  optim.lr=1e-4 \
  checkpointing.resume_from_ckpt=false \
  seed=42
