"""
Training Script for Hierarchy Predictor (Algorithm 1)

Trains the structure learning model on WikiHow dataset.
- Level 0: Summary tokens (less masked)
- Level 1: Content tokens (more masked)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Optional, Dict
import json

from wikihow_hierarchy_dataset import WikiHowHierarchyDataset
from hierarchical_noise_schedule import HierarchicalNoiseSchedule, HierarchicalMasker
from hierarchy_predictor import HierarchyPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchy Predictor")

    # Data arguments
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default="./data/wikihow")

    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrained_embeddings", type=str, default=None,
                        help="Path to pretrained embeddings (e.g., 'gpt2')")
    parser.add_argument("--freeze_token_embeddings", action="store_true")

    # Noise schedule arguments
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="cosine",
                        choices=["cosine", "linear", "sqrt"])
    parser.add_argument("--mask_token_id", type=int, default=50256)

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)

    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default="hierarchy-predictor")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/hierarchy_predictor")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def setup_wandb(args):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"hierarchy_predictor_h{args.hidden_size}_l{args.num_layers}",
        config=vars(args),
    )


def create_model(args) -> nn.Module:
    """Create the Hierarchy Predictor model."""
    model = HierarchyPredictor(
        vocab_size=args.vocab_size,
        num_levels=args.num_levels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_length=args.max_length,
        pretrained_embeddings=args.pretrained_embeddings,
        freeze_token_embeddings=args.freeze_token_embeddings,
    )
    return model


def create_noise_schedule(args):
    """Create hierarchical noise schedule and masker."""
    noise_schedule = HierarchicalNoiseSchedule(
        num_levels=args.num_levels,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
    )

    masker = HierarchicalMasker(
        noise_schedule=noise_schedule,
        mask_token_id=args.mask_token_id,
        num_levels=args.num_levels,
    )

    return noise_schedule, masker


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    masker: HierarchicalMasker,
    optimizer: optim.Optimizer,
    scheduler,
    args,
    device: str,
) -> Dict[str, float]:
    """Execute one training step following Algorithm 1."""
    model.train()

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    hierarchy_labels = batch['hierarchy_labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    batch_size = input_ids.shape[0]

    # Sample random timesteps
    timesteps = torch.randint(
        1, args.num_timesteps + 1, (batch_size,), device=device
    )

    # Apply hierarchical masking
    masked_output = masker(input_ids, hierarchy_labels, timesteps, attention_mask)

    # Forward pass
    logits = model(
        masked_output['masked_ids'],
        masked_output['hierarchy_input'],
        attention_mask,
    )

    # Compute loss on masked positions only
    loss = model.compute_loss(logits, hierarchy_labels, masked_output['mask'])

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    optimizer.step()
    scheduler.step()

    # Compute accuracy on masked positions
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        mask = masked_output['mask'].bool()
        if mask.sum() > 0:
            correct = (predictions[mask] == hierarchy_labels[mask]).float().mean()
            accuracy = correct.item()
        else:
            accuracy = 0.0

    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'lr': scheduler.get_last_lr()[0],
        'masked_tokens': mask.sum().item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    masker: HierarchicalMasker,
    args,
    device: str,
) -> Dict[str, float]:
    """Evaluate the model on validation set."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        hierarchy_labels = batch['hierarchy_labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        batch_size = input_ids.shape[0]

        # Use middle timestep for evaluation
        timesteps = torch.full((batch_size,), args.num_timesteps // 2, device=device)

        # Apply masking
        masked_output = masker(input_ids, hierarchy_labels, timesteps, attention_mask)

        # Forward pass
        logits = model(
            masked_output['masked_ids'],
            masked_output['hierarchy_input'],
            attention_mask,
        )

        # Compute loss
        loss = model.compute_loss(logits, hierarchy_labels, masked_output['mask'])
        total_loss += loss.item() * batch_size

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        mask = masked_output['mask'].bool()
        total_correct += (predictions[mask] == hierarchy_labels[mask]).sum().item()
        total_masked += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / max(1, total_masked)

    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    step: int,
    args,
    metrics: Dict[str, float],
):
    """Save model checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
        'metrics': metrics,
    }

    path = os.path.join(args.output_dir, f"checkpoint_step{step}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    # Also save latest
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup wandb
    setup_wandb(args)

    # Create datasets
    print("Loading datasets...")
    train_dataset = WikiHowHierarchyDataset(
        tokenizer_name="gpt2",
        max_length=args.max_length,
        split="train[:95%]",
        cache_dir=args.cache_dir,
    )

    valid_dataset = WikiHowHierarchyDataset(
        tokenizer_name="gpt2",
        max_length=args.max_length,
        split="train[95%:]",
        cache_dir=args.cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    # Create model
    print("Creating model...")
    model = create_model(args)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create noise schedule and masker
    noise_schedule, masker = create_noise_schedule(args)
    noise_schedule = noise_schedule.to(device)
    masker.noise_schedule = noise_schedule

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # Training loop
    print("Starting training...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            # Training step
            metrics = train_step(
                model, batch, masker, optimizer, scheduler, args, device
            )

            global_step += 1

            # Log to wandb
            wandb.log({
                'train/loss': metrics['loss'],
                'train/accuracy': metrics['accuracy'],
                'train/lr': metrics['lr'],
                'train/masked_tokens': metrics['masked_tokens'],
                'step': global_step,
            })

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
            })

            # Evaluate
            if global_step % args.eval_every == 0:
                val_metrics = evaluate(model, valid_loader, masker, args, device)
                print(f"\nStep {global_step}: val_loss={val_metrics['val_loss']:.4f}, "
                      f"val_acc={val_metrics['val_accuracy']:.3f}")

                wandb.log({
                    'val/loss': val_metrics['val_loss'],
                    'val/accuracy': val_metrics['val_accuracy'],
                    'step': global_step,
                })

                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, args,
                        {'best_val_loss': best_val_loss, **val_metrics}
                    )

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, args, metrics
                )

    # Final evaluation
    print("\nFinal evaluation...")
    val_metrics = evaluate(model, valid_loader, masker, args, device)
    print(f"Final: val_loss={val_metrics['val_loss']:.4f}, "
          f"val_acc={val_metrics['val_accuracy']:.3f}")

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, global_step, args,
        {'final': True, **val_metrics}
    )

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
