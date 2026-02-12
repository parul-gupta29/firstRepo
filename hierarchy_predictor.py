"""
Hierarchy Predictor Model (Algorithm 1)

Predicts hierarchy labels for masked tokens.
Uses hierarchy embeddings added to token embeddings.

Architecture:
    Input: masked tokens + hierarchy embeddings (K for unknown)
    Output: K-class classification per token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from transformers import GPT2Config, GPT2Model


class HierarchyEmbedding(nn.Module):
    """
    Learnable hierarchy embeddings.
    K levels + 1 for "unknown" (masked tokens).
    """

    def __init__(self, num_levels: int, hidden_size: int):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_size = hidden_size
        # K+1 embeddings: 0 to K-1 for known levels, K for unknown
        self.embedding = nn.Embedding(num_levels + 1, hidden_size)

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, hierarchy_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hierarchy_labels: [B, L] with values 0 to K (K = unknown)
        Returns:
            embeddings: [B, L, hidden_size]
        """
        return self.embedding(hierarchy_labels)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x


class HierarchyPredictor(nn.Module):
    """
    Model 1: Hierarchy Predictor for Structure Learning

    Takes masked tokens with hierarchy input embeddings and predicts
    the hierarchy level for each masked position.

    Architecture:
        - Token embedding layer (frozen or trainable)
        - Hierarchy embedding layer (learnable, added to token embeddings)
        - Transformer encoder
        - Classification head (predicts K classes per token)
    """

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        num_levels: int = 2,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 1024,
        pretrained_embeddings: Optional[str] = None,
        freeze_token_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_levels = num_levels
        self.hidden_size = hidden_size
        self.unknown_level = num_levels  # K = unknown

        # Token embeddings
        if pretrained_embeddings:
            # Load from pretrained model (e.g., GPT-2)
            from transformers import AutoModel
            pretrained = AutoModel.from_pretrained(pretrained_embeddings)
            self.token_embedding = pretrained.wte
            if hidden_size != self.token_embedding.embedding_dim:
                self.token_proj = nn.Linear(
                    self.token_embedding.embedding_dim, hidden_size
                )
            else:
                self.token_proj = nn.Identity()
        else:
            self.token_embedding = nn.Embedding(vocab_size, hidden_size)
            self.token_proj = nn.Identity()

        if freeze_token_embeddings:
            for param in self.token_embedding.parameters():
                param.requires_grad = False

        # Hierarchy embeddings (learnable)
        self.hierarchy_embedding = HierarchyEmbedding(num_levels, hidden_size)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_length, hidden_size)

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Classification head: predict hierarchy level for each token
        self.classifier = nn.Linear(hidden_size, num_levels)

    def forward(
        self,
        input_ids: torch.Tensor,
        hierarchy_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [B, L] - masked token IDs
            hierarchy_input: [B, L] - hierarchy labels (K for masked/unknown)
            attention_mask: [B, L] - attention mask (1 = attend, 0 = ignore)

        Returns:
            logits: [B, L, num_levels] - hierarchy predictions for each token
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        token_emb = self.token_proj(self.token_embedding(input_ids))
        hier_emb = self.hierarchy_embedding(hierarchy_input)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Combine: token + hierarchy + position
        x = token_emb + hier_emb + pos_emb
        x = self.dropout(x)

        # Convert attention mask for MultiheadAttention (True = ignore)
        if attention_mask is not None:
            attn_mask = (1 - attention_mask).bool()
        else:
            attn_mask = None

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.norm(x)

        # Classify hierarchy level
        logits = self.classifier(x)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        hierarchy_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss only on masked positions.

        Args:
            logits: [B, L, num_levels] - predicted logits
            hierarchy_labels: [B, L] - ground truth hierarchy levels
            mask: [B, L] - binary mask (1 = masked position, 0 = unmasked)

        Returns:
            loss: scalar loss value
        """
        # Only compute loss on masked positions
        masked_logits = logits[mask.bool()]  # [N, num_levels]
        masked_labels = hierarchy_labels[mask.bool()]  # [N]

        if masked_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(masked_logits, masked_labels)
        return loss

    def predict(
        self,
        input_ids: torch.Tensor,
        hierarchy_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict hierarchy levels with probabilities.

        Returns:
            dict with:
                - logits: [B, L, num_levels]
                - probs: [B, L, num_levels] - softmax probabilities
                - predictions: [B, L] - argmax predictions
        """
        logits = self.forward(input_ids, hierarchy_input, attention_mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
        }


class HierarchyPredictorLightning(nn.Module):
    """
    PyTorch Lightning wrapper for HierarchyPredictor.
    Handles training loop, validation, and logging.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = HierarchyPredictor(**config['model'])

        # Import here to avoid circular imports
        from hierarchical_noise_schedule import HierarchicalNoiseSchedule, HierarchicalMasker

        self.noise_schedule = HierarchicalNoiseSchedule(
            num_levels=config['model']['num_levels'],
            num_timesteps=config.get('num_timesteps', 1000),
            schedule_type=config.get('schedule_type', 'cosine'),
        )

        self.masker = HierarchicalMasker(
            noise_schedule=self.noise_schedule,
            mask_token_id=config.get('mask_token_id', 50256),  # GPT-2 EOS as mask
            num_levels=config['model']['num_levels'],
        )

    def forward(self, batch: dict, timestep: Optional[torch.Tensor] = None):
        """
        Training forward pass following Algorithm 1.
        """
        input_ids = batch['input_ids']
        hierarchy_labels = batch['hierarchy_labels']
        attention_mask = batch.get('attention_mask')

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Sample random timestep if not provided
        if timestep is None:
            timestep = torch.randint(
                1, self.noise_schedule.num_timesteps + 1,
                (batch_size,), device=device
            )

        # Apply hierarchical masking
        masked_output = self.masker(
            input_ids, hierarchy_labels, timestep, attention_mask
        )

        # Forward through model
        logits = self.model(
            masked_output['masked_ids'],
            masked_output['hierarchy_input'],
            attention_mask,
        )

        # Compute loss on masked positions
        loss = self.model.compute_loss(
            logits, hierarchy_labels, masked_output['mask']
        )

        return {
            'loss': loss,
            'logits': logits,
            'mask': masked_output['mask'],
        }


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 128
    vocab_size = 50257
    num_levels = 2

    model = HierarchyPredictor(
        vocab_size=vocab_size,
        num_levels=num_levels,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )

    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    hierarchy_input = torch.randint(0, num_levels + 1, (batch_size, seq_len))
    hierarchy_labels = torch.randint(0, num_levels, (batch_size, seq_len))
    mask = torch.randint(0, 2, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids, hierarchy_input)
    print(f"Logits shape: {logits.shape}")  # [B, L, num_levels]

    # Compute loss
    loss = model.compute_loss(logits, hierarchy_labels, mask)
    print(f"Loss: {loss.item():.4f}")

    # Test prediction
    output = model.predict(input_ids, hierarchy_input)
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Probs shape: {output['probs'].shape}")

    print("\nModel test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
