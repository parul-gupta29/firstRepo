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

from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata

# Allowlist all common OmegaConf classes used in checkpoints
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Metadata])


def _is_lora_checkpoint(state_dict):
    """Check if checkpoint contains LoRA-wrapped weights."""
    for key in state_dict:
        if 'base_model.model.' in key or 'lora_A' in key or 'lora_B' in key:
            return True
    return False


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load MDLM model from checkpoint, with automatic LoRA detection."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']['config']
    else:
        raise ValueError("Cannot find config in checkpoint")

    tokenizer = dataloader.get_tokenizer(config)
    state_dict = checkpoint.get('state_dict', {})

    if _is_lora_checkpoint(state_dict):
        print("Detected LoRA checkpoint — applying PEFT wrapper before loading weights.")
        from peft import LoraConfig, get_peft_model

        # Create base model (without loading state dict)
        model = diffusion.Diffusion(config=config, tokenizer=tokenizer)

        # Read LoRA config from checkpoint's saved config
        lora_cfg = config.get('lora', None)
        if lora_cfg and lora_cfg.get('enabled', False):
            target_modules = list(lora_cfg.target_modules)
            modules_to_save = list(lora_cfg.modules_to_save) if lora_cfg.get('modules_to_save', None) else None
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                bias='all',
            )
        else:
            # Fallback: infer LoRA config from state dict keys
            print("  No lora config in checkpoint — using defaults (r=8, alpha=16).")
            target_modules = []
            for key in state_dict:
                if 'lora_A' in key:
                    # e.g. backbone.base_model.model.blocks.0.attn_qkv.lora_A.default.weight
                    # extract the module name: attn_qkv
                    parts = key.split('.')
                    lora_idx = parts.index('lora_A')
                    module_name = parts[lora_idx - 1]
                    if module_name not in target_modules:
                        target_modules.append(module_name)
            modules_to_save = []
            for key in state_dict:
                if 'modules_to_save' in key:
                    parts = key.split('.')
                    ms_idx = parts.index('modules_to_save')
                    module_name = parts[ms_idx - 2]  # e.g. adaLN_modulation
                    if module_name not in modules_to_save:
                        modules_to_save.append(module_name)
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=target_modules,
                modules_to_save=modules_to_save if modules_to_save else None,
                bias='all',
            )
            print(f"  Inferred target_modules: {target_modules}")
            print(f"  Inferred modules_to_save: {modules_to_save}")

        model.backbone = get_peft_model(model.backbone, peft_config)
        model.ema = None  # EMA is disabled for LoRA

        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        print("LoRA model weights loaded.")
    else:
        model = diffusion.Diffusion.load_from_checkpoint(
            checkpoint_path,
            config=config,
            tokenizer=tokenizer,
            map_location=device,
            weights_only=False
        )

    model = model.to(device)
    model.eval()

    # Debug info
    try:
        v_size = getattr(config.data, 'vocab_size', getattr(config.data, 'n_tokens', "Unknown"))
        s_len = getattr(config.model, 'length', getattr(config.model, 'seq_len', "Unknown"))
        print(f"Model loaded successfully!")
        print(f"Vocab size: {v_size}")
        print(f"Sequence length: {s_len}")
    except Exception as e:
        print(f"Model loaded, but could not read all config keys: {e}")
        print("Available data keys:", config.data.keys())

    return model, tokenizer, config


def sample_with_prompt(model, tokenizer, prompts, num_steps=128, eps=1e-5):
    """
    Generate text conditioned on prompts using MDLM infilling.

    UPDATED: Includes explicit truncation at EOS token to prevent "zombie" generations.
    """
    device = model.device
    batch_size = len(prompts)
    seq_len = model.config.model.length
    print("sequence length: ", seq_len)
    mask_index = model.mask_index

    # --- NEW: Get EOS token ID for truncation ---
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    else:
        # Fallback for GPT-2 if not explicitly set (usually 50256)
        eos_token_id = 50256

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

    # --- UPDATED DECODING WITH EOS TRUNCATION ---
    text_samples = []
    # Iterate through the batch to cut off at EOS
    for i in range(batch_size):
        seq = x[i]
        # Find all indices where the token is EOS
        # GPT-2 EOS is typically 50256 (<|endoftext|>)
        eos_indices = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            # Cut the sequence at the *first* EOS token
            first_eos_idx = eos_indices[0].item()
            seq = seq[:first_eos_idx]
        # Decode the clean sequence
        decoded_text = tokenizer.decode(seq, skip_special_tokens=True)
        text_samples.append(decoded_text)

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
    Create evaluation prompts from WikiHow HELD-OUT test set.
    Uses the first sentence of each article as the prompt.

    Data splits:
      - train[:90%]    = Training data
      - train[90%:95%] = Validation data
      - train[95%:]    = Test data (HELD-OUT, never seen during training)
    """
    import datasets

    print(f"Loading WikiHow HELD-OUT test set for evaluation prompts...")
    print(f"  (train[95%:] - never seen during training or validation)")
    dataset = datasets.load_dataset(
        'gursi26/wikihow-cleaned',
        split='train[95%:]',  # Held-out test set (never seen during training)
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
