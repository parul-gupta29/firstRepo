#!/usr/bin/env python
"""
Inference script for MDLM model fine-tuned on WikiHow.

Usage:
    python run_inference.py --checkpoint_path /path/to/best.ckpt --num_samples 5
"""

import argparse
import torch
import hydra
from omegaconf import OmegaConf
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataloader
import diffusion


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load MDLM model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']['config']
    else:
        raise ValueError("Cannot find config in checkpoint")

    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)

    # Load model
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        config=config,
        tokenizer=tokenizer,
        map_location=device
    )
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Vocab size: {model.config.data.vocab_size}")
    print(f"Sequence length: {model.config.model.length}")

    return model, tokenizer, config


def generate_samples(model, tokenizer, num_samples=5, num_steps=128):
    """Generate text samples from the model."""
    print(f"\nGenerating {num_samples} samples with {num_steps} diffusion steps...")

    # Store original batch size and temporarily modify
    original_eval_batch_size = model.config.loader.eval_batch_size
    model.config.loader.eval_batch_size = num_samples

    samples_list = []

    with torch.no_grad():
        # Use EMA weights if available
        if model.ema:
            model.ema.store(model.backbone.parameters())
            model.ema.copy_to(model.backbone.parameters())

        # Generate samples
        samples = model._sample(num_steps=num_steps, eps=1e-5)

        # Restore original weights
        if model.ema:
            model.ema.restore(model.backbone.parameters())

    # Decode samples
    text_samples = tokenizer.batch_decode(samples, skip_special_tokens=False)

    # Restore original batch size
    model.config.loader.eval_batch_size = original_eval_batch_size

    return text_samples


def main():
    parser = argparse.ArgumentParser(description='Generate samples from fine-tuned MDLM')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint (best.ckpt)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='Number of diffusion steps for sampling')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Optional file to save generated samples')
    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)

    # Load model
    model, tokenizer, config = load_model_from_checkpoint(
        args.checkpoint_path,
        device=args.device
    )

    # Generate samples
    samples = generate_samples(
        model,
        tokenizer,
        num_samples=args.num_samples,
        num_steps=args.num_steps
    )

    # Print samples
    print("\n" + "="*80)
    print("GENERATED SAMPLES")
    print("="*80)

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        # Clean up the sample for display
        clean_sample = sample.replace('<|endoftext|>', '\n[END]')
        print(clean_sample[:2000])  # Limit output length
        if len(sample) > 2000:
            print("... [truncated]")

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for i, sample in enumerate(samples):
                f.write(f"--- Sample {i+1} ---\n")
                f.write(sample)
                f.write("\n\n")
        print(f"\nSamples saved to: {args.output_file}")


if __name__ == '__main__':
    main()
