#!/usr/bin/env python
"""
Inference and evaluation script for MDLM model fine-tuned on WikiHow.

Supports:
- Unconditional generation
- Prompt-conditioned generation (infilling)
- Batch evaluation from prompt file
- Output in JSON format for perplexity/MAUVE evaluation

Usage:
    # Unconditional generation
    python run_inference.py --checkpoint_path /path/to/best.ckpt --num_samples 5

    # Prompt-conditioned generation
    python run_inference.py --checkpoint_path /path/to/best.ckpt --prompt "How to bake a cake"

    # Batch evaluation from file
    python run_inference.py --checkpoint_path /path/to/best.ckpt --prompt_file prompts.txt --output_file results.json
"""

import argparse
import json
import torch
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataloader
import diffusion


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load MDLM model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']['config']
    else:
        raise ValueError("Cannot find config in checkpoint")

    tokenizer = dataloader.get_tokenizer(config)

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


def sample_with_prompt(model, tokenizer, prompts, num_steps=128, eps=1e-5):
    """
    Generate text conditioned on prompts using MDLM infilling.

    Args:
        model: MDLM diffusion model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        num_steps: Number of diffusion steps
        eps: Small epsilon for numerical stability

    Returns:
        List of generated text strings
    """
    device = model.device
    batch_size = len(prompts)
    seq_len = model.config.model.length
    mask_index = model.mask_index

    # Tokenize prompts
    prompt_tokens_list = []
    prompt_lengths = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        # Truncate if too long (leave room for generation)
        max_prompt_len = seq_len // 2
        if len(tokens) > max_prompt_len:
            tokens = tokens[:max_prompt_len]
        prompt_tokens_list.append(tokens)
        prompt_lengths.append(len(tokens))

    # Initialize: prompt tokens + masked tokens for the rest
    x = torch.full((batch_size, seq_len), mask_index, dtype=torch.int64, device=device)
    prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    for i, tokens in enumerate(prompt_tokens_list):
        x[i, :len(tokens)] = torch.tensor(tokens, device=device)
        prompt_mask[i, :len(tokens)] = True

    # Diffusion sampling with prompt conditioning
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    with torch.no_grad():
        # Use EMA weights if available
        if model.ema:
            model.ema.store(model.backbone.parameters())
            model.ema.copy_to(model.backbone.parameters())

        for i in tqdm(range(num_steps), desc="Sampling"):
            t = timesteps[i] * torch.ones(batch_size, 1, device=device)

            if model.sampler == 'ddpm':
                sigma_t, _ = model.noise(t)
                sigma_s, _ = model.noise(t - dt)
                if sigma_t.ndim > 1:
                    sigma_t = sigma_t.squeeze(-1)
                if sigma_s.ndim > 1:
                    sigma_s = sigma_s.squeeze(-1)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_s = 1 - torch.exp(-sigma_s)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_s = move_chance_s[:, None, None]

                log_p_x0 = model.forward(x, sigma_t)
                q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
                q_xs[:, :, mask_index] = move_chance_s[:, :, 0]

                # Sample new tokens
                gumbel_norm = 1e-10 - (torch.rand_like(q_xs) + 1e-10).log()
                _x = (q_xs / gumbel_norm).argmax(dim=-1)

                # Keep non-masked tokens and prompt tokens fixed
                copy_flag = ((x != mask_index) | prompt_mask).to(x.dtype)
                x = (copy_flag * x + (1 - copy_flag) * _x).long()

            elif model.sampler == 'ddpm_cache':
                sigma_t, _ = model.noise(t)
                if t.ndim > 1:
                    t_squeezed = t.squeeze(-1)
                else:
                    t_squeezed = t
                move_chance_t = t_squeezed[:, None, None]
                move_chance_s = (t_squeezed - dt)[:, None, None]

                if p_x0_cache is None:
                    p_x0_cache = model.forward(x, sigma_t).exp()

                q_xs = p_x0_cache * (move_chance_t - move_chance_s)
                q_xs[:, :, mask_index] = move_chance_s[:, :, 0]

                gumbel_norm = 1e-10 - (torch.rand_like(q_xs) + 1e-10).log()
                _x = (q_xs / gumbel_norm).argmax(dim=-1)

                copy_flag = ((x != mask_index) | prompt_mask).to(x.dtype)
                x_next = (copy_flag * x + (1 - copy_flag) * _x).long()

                if not torch.allclose(x_next, x) or model.time_conditioning:
                    p_x0_cache = None
                x = x_next

        # Final denoising step
        if model.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(batch_size, 1, device=device)
            unet_conditioning = model.noise(t)[0]
            logits = model.forward(x, unet_conditioning)
            _x = logits.argmax(dim=-1)
            # Keep prompt fixed
            copy_flag = prompt_mask.to(x.dtype)
            x = (copy_flag * x + (1 - copy_flag) * _x).long()

        # Restore original weights
        if model.ema:
            model.ema.restore(model.backbone.parameters())

    # Decode
    text_samples = tokenizer.batch_decode(x, skip_special_tokens=False)
    return text_samples, prompt_lengths


def generate_unconditional(model, tokenizer, num_samples=5, num_steps=128):
    """Generate unconditional samples."""
    print(f"\nGenerating {num_samples} unconditional samples...")

    original_batch_size = model.config.loader.eval_batch_size
    model.config.loader.eval_batch_size = num_samples

    with torch.no_grad():
        if model.ema:
            model.ema.store(model.backbone.parameters())
            model.ema.copy_to(model.backbone.parameters())

        samples = model._sample(num_steps=num_steps, eps=1e-5)

        if model.ema:
            model.ema.restore(model.backbone.parameters())

    text_samples = tokenizer.batch_decode(samples, skip_special_tokens=False)
    model.config.loader.eval_batch_size = original_batch_size

    return text_samples


def load_prompts_from_file(filepath):
    """Load prompts from a text file (one prompt per line) or JSON file."""
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], dict):
                    return [item.get('prompt', item.get('text', '')) for item in data]
                return data
            return [data]
    else:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def create_evaluation_prompts_from_wikihow(num_prompts=100, cache_dir=None):
    """
    Create evaluation prompts from WikiHow test set.
    Uses the first sentence of each article as the prompt.
    """
    import datasets

    print(f"Loading WikiHow dataset for evaluation prompts...")
    dataset = datasets.load_dataset(
        'ajibawa-2023/WikiHow',
        split='train[-5%:]',  # Use last 5% as test
        cache_dir=cache_dir
    )

    prompts = []
    references = []

    for i, item in enumerate(dataset):
        if i >= num_prompts:
            break

        text = item.get('text', '')
        if not text:
            continue

        # Split into sentences and use first as prompt
        sentences = text.split('. ')
        if len(sentences) >= 2:
            prompt = sentences[0] + '.'
            reference = text
            prompts.append(prompt)
            references.append(reference)

    return prompts, references


def save_results_json(results, output_file):
    """Save results in JSON format for evaluation."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate samples from fine-tuned MDLM')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples for unconditional generation')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file (supports .json and .txt)')

    # Prompt-conditioned generation
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt for conditioned generation')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='File with prompts (one per line or JSON)')

    # Evaluation mode
    parser.add_argument('--eval_mode', action='store_true',
                        help='Run evaluation on WikiHow test set')
    parser.add_argument('--num_eval_prompts', type=int, default=100,
                        help='Number of evaluation prompts from WikiHow')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)

    model, tokenizer, config = load_model_from_checkpoint(
        args.checkpoint_path, device=args.device
    )

    results = {
        'model': 'mdlm-wikihow',
        'checkpoint': args.checkpoint_path,
        'num_steps': args.num_steps,
        'samples': []
    }

    # Mode 1: Evaluation on WikiHow test set
    if args.eval_mode:
        print("\n=== Evaluation Mode ===")
        prompts, references = create_evaluation_prompts_from_wikihow(
            num_prompts=args.num_eval_prompts
        )
        print(f"Loaded {len(prompts)} evaluation prompts")

        all_generations = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            generations, prompt_lens = sample_with_prompt(
                model, tokenizer, batch_prompts, num_steps=args.num_steps
            )
            all_generations.extend(generations)

        for i, (prompt, gen, ref) in enumerate(zip(prompts, all_generations, references)):
            results['samples'].append({
                'id': i,
                'prompt': prompt,
                'generation': gen,
                'reference': ref
            })

    # Mode 2: Prompt-conditioned generation
    elif args.prompt or args.prompt_file:
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = load_prompts_from_file(args.prompt_file)

        print(f"\n=== Prompt-Conditioned Generation ({len(prompts)} prompts) ===")

        all_generations = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            generations, prompt_lens = sample_with_prompt(
                model, tokenizer, batch_prompts, num_steps=args.num_steps
            )
            all_generations.extend(generations)

        for i, (prompt, gen) in enumerate(zip(prompts, all_generations)):
            results['samples'].append({
                'id': i,
                'prompt': prompt,
                'generation': gen
            })

    # Mode 3: Unconditional generation
    else:
        print("\n=== Unconditional Generation ===")
        samples = generate_unconditional(
            model, tokenizer, num_samples=args.num_samples, num_steps=args.num_steps
        )
        for i, sample in enumerate(samples):
            results['samples'].append({
                'id': i,
                'prompt': None,
                'generation': sample
            })

    # Print samples
    print("\n" + "=" * 80)
    print("GENERATED SAMPLES")
    print("=" * 80)

    for item in results['samples'][:5]:  # Show first 5
        print(f"\n--- Sample {item['id'] + 1} ---")
        if item['prompt']:
            print(f"PROMPT: {item['prompt']}")
        gen = item['generation'].replace('<|endoftext|>', '\n[END]')
        print(f"GENERATION: {gen[:1500]}")
        if len(gen) > 1500:
            print("... [truncated]")

    if len(results['samples']) > 5:
        print(f"\n... and {len(results['samples']) - 5} more samples")

    # Save results
    if args.output_file:
        if args.output_file.endswith('.json'):
            save_results_json(results, args.output_file)
        else:
            with open(args.output_file, 'w') as f:
                for item in results['samples']:
                    if item['prompt']:
                        f.write(f"PROMPT: {item['prompt']}\n")
                    f.write(f"GENERATION: {item['generation']}\n")
                    f.write("\n---\n\n")
            print(f"Results saved to: {args.output_file}")

    # Print format info for evaluation
    print("\n" + "=" * 80)
    print("EVALUATION INFO")
    print("=" * 80)
    print("""
For MAUVE score calculation:
    from mauve import compute_mauve
    results = compute_mauve(
        p_text=[s['reference'] for s in data['samples']],  # human text
        q_text=[s['generation'] for s in data['samples']], # model text
        device_id=0,
        max_text_length=512
    )
    print(f"MAUVE score: {results.mauve}")

For Perplexity (using GPT-2):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    # Compute perplexity on generations
""")


if __name__ == '__main__':
    main()
