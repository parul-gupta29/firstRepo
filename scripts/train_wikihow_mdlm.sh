#!/bin/bash
#SBATCH --job-name=mdlm-wikihow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err

# Activate conda environment
source ~/.bashrc
conda activate mdlm

# Create necessary directories
mkdir -p outputs
mkdir -p watch_folder
mkdir -p data/wikihow

# Train MDLM on WikiHow dataset
python main.py \
  model=small \
  data=wikihow \
  wandb.name=mdlm-wikihow \
  parameterization=subs \
  model.length=1024 \
  loader.global_batch_size=256 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.max_steps=500000 \
  trainer.val_check_interval=5000 \
  seed=42
