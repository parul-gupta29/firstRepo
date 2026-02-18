import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _is_lora_checkpoint(state_dict):
  """Check if checkpoint contains LoRA-wrapped weights."""
  for key in state_dict:
    if 'base_model.model.' in key or 'lora_A' in key or 'lora_B' in key:
      return True
  return False


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  checkpoint = torch.load(
    config.eval.checkpoint_path,
    map_location='cuda',
    weights_only=False)
  state_dict = checkpoint.get('state_dict', {})

  if _is_lora_checkpoint(state_dict):
    from peft import LoraConfig, get_peft_model

    model = diffusion.Diffusion(
      config=config, tokenizer=tokenizer)

    # Read LoRA config from checkpoint's saved hyperparameters
    # (must match the rank/alpha used during training)
    ckpt_config = checkpoint.get(
      'hyper_parameters', {}).get('config', config)
    lora_cfg = ckpt_config.get('lora', None)
    if lora_cfg and lora_cfg.get('enabled', False):
      target_modules = list(lora_cfg.target_modules)
      modules_to_save = (
        list(lora_cfg.modules_to_save)
        if lora_cfg.get('modules_to_save', None) else None)
      peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias='all')
    else:
      # Fallback: infer LoRA config from state dict keys
      target_modules = []
      for key in state_dict:
        if 'lora_A' in key:
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
          module_name = parts[ms_idx - 2]
          if module_name not in modules_to_save:
            modules_to_save.append(module_name)
      peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=target_modules,
        modules_to_save=modules_to_save if modules_to_save
          else None,
        bias='all')

    model.backbone = get_peft_model(
      model.backbone, peft_config)
    model.ema = None
    model.load_state_dict(state_dict, strict=False)
    return model

  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    weights_only=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  # Load pretrained backbone weights for fine-tuning
  pretrained_model_name = config.get('pretrained_model_name', None)
  if pretrained_model_name:
    logger.info(
      f'Loading pretrained backbone from: {pretrained_model_name}')
    from huggingface_hub import hf_hub_download
    try:
      weights_path = hf_hub_download(
        pretrained_model_name, 'model.safetensors')
      from safetensors.torch import load_file
      pretrained_state_dict = load_file(weights_path)
    except Exception:
      weights_path = hf_hub_download(
        pretrained_model_name, 'pytorch_model.bin')
      pretrained_state_dict = torch.load(
        weights_path, map_location='cpu')
    # Strip 'backbone.' prefix from keys since we load into model.backbone
    backbone_state_dict = {}
    for k, v in pretrained_state_dict.items():
      if k.startswith('backbone.'):
        backbone_state_dict[k[len('backbone.'):]] = v
      else:
        backbone_state_dict[k] = v
    del pretrained_state_dict
    missing, unexpected = model.backbone.load_state_dict(
      backbone_state_dict, strict=False)
    if missing:
      logger.warning(f'Missing keys when loading pretrained: {missing}')
    if unexpected:
      logger.warning(f'Unexpected keys when loading pretrained: {unexpected}')
    del backbone_state_dict
    logger.info('Pretrained backbone loaded successfully.')

    # Reinitialize EMA with pretrained weights (EMA was initialized
    # with random params before pretrained loading)
    if model.ema is not None:
      logger.info('Reinitializing EMA with pretrained weights.')
      import itertools
      import models.ema as ema_module
      model.ema = ema_module.ExponentialMovingAverage(
        itertools.chain(model.backbone.parameters(),
                        model.noise.parameters()),
        decay=config.training.ema)

  # Apply LoRA/QLoRA if configured
  lora_config = config.get('lora', None)
  if lora_config and lora_config.get('enabled', False):
    from peft import LoraConfig, get_peft_model
    logger.info('Applying LoRA to backbone.')

    target_modules = list(lora_config.target_modules)
    modules_to_save = list(lora_config.modules_to_save) if lora_config.get('modules_to_save', None) else None
    peft_config = LoraConfig(
      r=lora_config.r,
      lora_alpha=lora_config.alpha,
      lora_dropout=lora_config.dropout,
      target_modules=target_modules,
      modules_to_save=modules_to_save,
      bias='all',
    )
    model.backbone = get_peft_model(model.backbone, peft_config)

    model.backbone.print_trainable_parameters()

    # Disable EMA when using LoRA (EMA was initialized with pre-LoRA params)
    if model.ema is not None:
      logger.info('Disabling EMA for LoRA fine-tuning.')
      model.ema = None

    # Override global_batch_size for LoRA fine-tuning
    if lora_config.get('global_batch_size', None):
      from omegaconf import OmegaConf, open_dict
      lora_gbs = lora_config.global_batch_size
      OmegaConf.update(config, 'loader.global_batch_size', lora_gbs)
      # Recompute accumulate_grad_batches since Hydra resolvers
      # already ran with the original global_batch_size
      num_devices = config.trainer.devices
      num_nodes = config.trainer.num_nodes
      per_device_bs = config.loader.batch_size
      new_accum = max(1, lora_gbs // (num_devices * per_device_bs * num_nodes))
      with open_dict(config):
        config.trainer.accumulate_grad_batches = new_accum
      logger.info(
        f'LoRA: overriding global_batch_size to {lora_gbs}, '
        f'accumulate_grad_batches to {new_accum}')

    # Override LR scheduler for LoRA fine-tuning
    if lora_config.get('lr_scheduler', None):
      from omegaconf import OmegaConf, DictConfig, open_dict
      new_scheduler = DictConfig({
        '_target_': 'utils.CosineDecayWarmupLRScheduler',
        't_in_epochs': False,
        't_initial': config.trainer.max_steps - lora_config.warmup_steps,
        'warmup_prefix': True,
        'warmup_lr_init': 1e-6,
        'warmup_t': lora_config.warmup_steps,
        'lr_min': 1e-6,
      })
      with open_dict(config):
        config.lr_scheduler = new_scheduler
      logger.info('LoRA: using cosine decay LR scheduler')

    logger.info('LoRA applied successfully.')

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()