# -*- coding: utf-8 -*-

"""
dataloader.py: Create dataloaders and collate batches
"""

### IMPORTS ###
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizer


### CONSTANTS ###
BATCH_SIZE = 12000
ENGLISH_IDENTIFIER = "en"
GERMAN_IDENTIFIER = "de"
MAX_LENGTH = 128
PADDING_ID = 2
#TRANSLATION_IDENTIFIER = "translation"

def collate_batch(
    batch: list[dict[str, str]],
    src_tok: PreTrainedTokenizer,
    tgt_tok: PreTrainedTokenizer,
    device: torch.device,
    max_len: int, 
    pad_id: int=PADDING_ID) -> tuple[Tensor, Tensor]:
    src_batch, tgt_batch = [], []

    for item in batch:
        src_text = item[GERMAN_IDENTIFIER]
        tgt_text = item[ENGLISH_IDENTIFIER]

        src_enc = src_tok(src_text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        tgt_enc = tgt_tok(tgt_text, truncation=True, max_length=max_len, padding='max_length', return_tensors="pt")

        src_batch.append(src_enc["input_ids"].squeeze(0).to(device))
        tgt_batch.append(tgt_enc["input_ids"].squeeze(0).to(device))
    
    return torch.stack(src_batch), torch.stack(tgt_batch)

def create_dataloaders(
    device: torch.device,
    src_tok: PreTrainedTokenizer,
    tgt_tok: PreTrainedTokenizer,
    train_ds: Dataset,
    valid_ds: Dataset, 
    batch_size: int=BATCH_SIZE,
    max_len: int=MAX_LENGTH, 
    is_distributed=False,
) -> tuple[DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for training and validation datasets.

    This function constructs DataLoaders for machine translation data formatted
    as a list of JSON-like records. Each dataset element is expected to have
    the structure::

        {
            "translation": {
                "de": "<source sentence>",
                "en": "<target sentence>"
            }
        }

    Sentences are tokenized using HuggingFace tokenizers and padded dynamically
    within each batch. The returned DataLoaders support both single-GPU and
    distributed (multi-GPU) training depending on the value of
    ``is_distributed``.

    Args:
        device:
            Device on which the collated batches will be placed. Typically
            ``torch.device("cuda")`` or ``torch.device("cpu")``.
        src_tok:
            HuggingFace tokenizer for the source language (German).
        tgt_tok:
            HuggingFace tokenizer for the target language (English).
        train_ds:
            Training dataset represented as a list of translation samples.
        valid_ds:
            Validation dataset in the same format as ``train_ds``.
        batch_size:
            Number of samples per batch before padding. Defaults to ``12000``.
        max_len:
            Maximum allowed tokenized sequence length for truncation. Defaults
            to ``128``.
        is_distributed:
            Whether to enable ``DistributedSampler`` for multi-GPU training.
            Defaults to ``False``.

    Returns:
            A tuple ``(train_loader, valid_loader)`` where both elements are
            PyTorch ``DataLoader`` objects configured with dynamic padding and
            HuggingFace tokenization.
    """
    def collate_fn(batch: list[dict[str, str]]) -> tuple[Tensor, Tensor]:
        """Wrapper function used by ``DataLoader`` to collate samples into batches.

        Args:
            batch: A batch of translation samples, where each sample contains German
                and English sentences under the ``"translation"`` key.

        Returns:
            A tuple ``(src_batch, tgt_batch)`` produced by :func:`collate_batch`,
            containing padded token-ID tensors for source and target sequences.
        """
        return collate_batch(
            batch=batch,
            src_tok=src_tok,
            tgt_tok=tgt_tok,
            device=device,
            max_len=max_len,
            pad_id=PADDING_ID,
        ) 
    
    train_sampler = None
    valid_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(train_ds)
        valid_sampler = DistributedSampler(valid_ds)

    train_dataloader = DataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        dataset=valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    print("Sucessfully created train_dataloader and validation_dataloader.")

    return train_dataloader, valid_dataloader

if __name__ == '__main__':
    from dataset import load_multi30k
    from tokenizer import load_tokenizers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, valid_ds = load_multi30k()
    src_tok, tgt_tok = load_tokenizers()
    train_loader, valid_loader = create_dataloaders(
        device,
        src_tok,
        tgt_tok,
        train_ds,
        valid_ds,
    )

    sample = next(iter(train_loader))
    print(sample)