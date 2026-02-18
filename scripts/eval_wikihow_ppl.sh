#!/bin/bash
# Evaluate perplexity of a fine-tuned MDLM model on the WikiHow test set.
#
# This computes the diffusion-based perplexity (ELBO) on the held-out WikiHow
# test split (5% of the dataset, never seen during training), analogous to
# zero-shot perplexity evaluation on PubMed / WikiText / OpenWebText.
#
# Usage:
#   bash scripts/eval_wikihow_ppl.sh <checkpoint_path>
#
# Example:
#   bash scripts/eval_wikihow_ppl.sh ./outputs/wikihow-train/2025.01.01/120000/checkpoints/last.ckpt
#
# The script evaluates with both continuous time (T=0) and discrete time (T=1000).

if [ -z "$1" ]; then
  echo "Usage: bash scripts/eval_wikihow_ppl.sh <checkpoint_path>"
  echo "Example: bash scripts/eval_wikihow_ppl.sh ./checkpoints/last.ckpt"
  exit 1
fi

checkpoint_path=$1

# Create data cache directory
mkdir -p data/wikihow

export HYDRA_FULL_ERROR=1

for T in 0 1000; do
  echo "=== Evaluating WikiHow test set perplexity with T=$T ==="
  python main.py \
    mode=ppl_eval \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    data=wikihow_test \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    T="$T" \
    eval.checkpoint_path=$checkpoint_path \
    +wandb.offline=true
done
