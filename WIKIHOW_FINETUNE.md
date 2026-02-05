# Fine-tuning MDLM on WikiHow Dataset

This guide explains how to fine-tune the Masked Diffusion Language Model (MDLM) on the WikiHow dataset.

## Dataset

We use the [ajibawa-2023/WikiHow](https://huggingface.co/datasets/ajibawa-2023/WikiHow) dataset from HuggingFace, which contains 179k WikiHow articles suitable for text generation tasks.

The dataset is automatically split:
- **Training**: 95% of the data (~170k articles)
- **Validation**: 5% of the data (~9k articles)

## Setup

### 1. Create Conda Environment

```bash
conda env create -f requirements.yaml
conda activate mdlm
```

### 2. Create Required Directories

```bash
mkdir -p outputs
mkdir -p watch_folder
mkdir -p data/wikihow
```

## Training Options

### Option 1: Train from Scratch

Train MDLM from scratch on WikiHow:

```bash
python main.py \
  model=small \
  data=wikihow \
  wandb.name=mdlm-wikihow \
  parameterization=subs \
  model.length=1024 \
  loader.global_batch_size=256 \
  loader.batch_size=8 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  trainer.max_steps=500000
```

Or use the provided SLURM script:

```bash
sbatch scripts/train_wikihow_mdlm.sh
```

### Option 2: Fine-tune from Pretrained Checkpoint

For faster convergence, fine-tune from the pretrained OpenWebText model:

```bash
python main.py \
  mode=train \
  model=small \
  data=wikihow \
  wandb.name=mdlm-wikihow-finetune \
  parameterization=subs \
  model.length=1024 \
  loader.global_batch_size=128 \
  loader.batch_size=4 \
  optim.lr=1e-4 \
  trainer.max_steps=100000
```

Or use the provided script:

```bash
bash scripts/finetune_wikihow_mdlm.sh
```

## Configuration

Key parameters you may want to adjust:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.length` | Sequence length | 1024 |
| `loader.batch_size` | Per-GPU batch size | 8 |
| `loader.global_batch_size` | Total batch size | 256 |
| `optim.lr` | Learning rate | 3e-4 (1e-4 for fine-tuning) |
| `trainer.max_steps` | Training steps | 500000 |
| `sampling.steps` | Diffusion sampling steps | 1000 |

### Memory Considerations

If you encounter OOM errors, try:
- Reducing `loader.batch_size`
- Reducing `model.length` (e.g., 512 instead of 1024)
- Using gradient accumulation (will happen automatically if batch_size * num_gpus < global_batch_size)

## Evaluation

### Compute Perplexity

```bash
python main.py \
  mode=ppl_eval \
  data=wikihow \
  model=small \
  parameterization=subs \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  +wandb.offline=true
```

### Generate Samples

```bash
python main.py \
  mode=sample_eval \
  data=wikihow \
  model=small \
  parameterization=subs \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10
```

## Model Architecture

The default `small` model has:
- 12 transformer layers
- 768 hidden dimension
- 12 attention heads
- ~110M parameters

For larger models, use `model=medium` or create custom configs in `configs/model/`.

## References

- [MDLM Paper](https://arxiv.org/abs/2406.07524) (NeurIPS 2024)
- [MDLM GitHub](https://github.com/kuleshov-group/mdlm)
- [WikiHow Dataset](https://huggingface.co/datasets/ajibawa-2023/WikiHow)
