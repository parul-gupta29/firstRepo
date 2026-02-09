"""
WikiHow Hierarchy Dataset
- Level 0: Summary tokens (less noisy)
- Level 1: Content tokens (more noisy)
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional


class WikiHowHierarchyDataset(Dataset):
    """
    WikiHow dataset with hierarchy labels.
    Each sample contains:
        - input_ids: tokenized text (summary + content)
        - hierarchy_labels: 0 for summary tokens, 1 for content tokens
        - attention_mask: standard attention mask
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        split: str = "train[:95%]",
        cache_dir: str = "./data/wikihow",
        separator_token: str = " [SEP] ",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.separator_token = separator_token

        # Load WikiHow dataset
        print(f"Loading WikiHow dataset (split: {split})...")
        self.dataset = load_dataset(
            "gursi26/wikihow-cleaned",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"Loaded {len(self.dataset)} samples")

        # Preprocess and cache
        self.processed_data = self._preprocess_all()

    def _preprocess_all(self) -> List[Dict]:
        """Preprocess all samples with hierarchy labels."""
        processed = []

        for idx, item in enumerate(self.dataset):
            result = self._preprocess_single(item)
            if result is not None:
                processed.append(result)

            if (idx + 1) % 10000 == 0:
                print(f"Preprocessed {idx + 1}/{len(self.dataset)} samples")

        print(f"Final dataset size: {len(processed)} samples")
        return processed

    def _preprocess_single(self, item: Dict) -> Optional[Dict]:
        """
        Preprocess a single WikiHow item.

        Structure:
            [BOS] <summary tokens> [SEP] <content tokens> [EOS] [PAD...]

        Hierarchy labels:
            - 0 for summary tokens (including BOS, SEP)
            - 1 for content tokens (including EOS)
        """
        summary = item.get('summary', '') or ''
        content = item.get('text', '') or ''

        # Skip empty samples
        if not summary.strip() or not content.strip():
            return None

        # Tokenize summary and content separately
        summary_tokens = self.tokenizer.encode(
            summary.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length // 4  # Reserve space for content
        )

        content_tokens = self.tokenizer.encode(
            content.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - len(summary_tokens) - 3  # BOS, SEP, EOS
        )

        # Skip if either is empty after tokenization
        if len(summary_tokens) == 0 or len(content_tokens) == 0:
            return None

        # Build full sequence: [BOS] summary [SEP] content [EOS]
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id or 0
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id or 0
        sep_id = self.tokenizer.sep_token_id or eos_id

        input_ids = [bos_id] + summary_tokens + [sep_id] + content_tokens + [eos_id]

        # Create hierarchy labels
        # Level 0: BOS + summary + SEP
        # Level 1: content + EOS
        hierarchy_labels = (
            [0] * (1 + len(summary_tokens) + 1) +  # BOS + summary + SEP = level 0
            [1] * (len(content_tokens) + 1)         # content + EOS = level 1
        )

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            hierarchy_labels = hierarchy_labels[:self.max_length]

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        hierarchy_labels = hierarchy_labels + [0] * padding_length  # Pad with level 0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'hierarchy_labels': torch.tensor(hierarchy_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'summary_length': 1 + len(summary_tokens) + 1,  # BOS + summary + SEP
            'content_length': len(content_tokens) + 1,       # content + EOS
        }

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]


def create_dataloaders(
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_dir: str = "./data/wikihow",
):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader

    train_dataset = WikiHowHierarchyDataset(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train[:95%]",
        cache_dir=cache_dir,
    )

    valid_dataset = WikiHowHierarchyDataset(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train[95%:]",
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = WikiHowHierarchyDataset(
        max_length=512,
        split="train[:1000]",  # Small subset for testing
    )

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Hierarchy labels shape: {sample['hierarchy_labels'].shape}")
    print(f"Unique hierarchy labels: {sample['hierarchy_labels'].unique()}")
    print(f"Summary length: {sample['summary_length']}")
    print(f"Content length: {sample['content_length']}")
