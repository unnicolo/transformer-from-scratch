# -*- coding: utf-8 -*-

"""
dataloader.py: Create dataloaders and collate batches
"""

### IMPORTS ###
import torch
from torch import device, Tensor
from torch.nn.functional import pad
from transformers import PreTrainedTokenizer


### CONSTANTS ###
TRANSLATION_IDENTIFIER = "translation"
GERMAN_IDENTIFIER = "de"
ENGLISH_IDENTIFIER = "en"
PADDING_ID = 2

def collate_batch(
    batch: list[dict[str, dict[str, str]]],
    src_tok: PreTrainedTokenizer,
    tgt_tok: PreTrainedTokenizer,
    device: device,
    max_len: int, 
    pad_id: int=PADDING_ID) -> tuple[Tensor, Tensor]:
    src_batch, tgt_batch = [], []

    for item in batch:
        src_text = item[TRANSLATION_IDENTIFIER][GERMAN_IDENTIFIER]
        tgt_text = item[TRANSLATION_IDENTIFIER][ENGLISH_IDENTIFIER]

        src_ids = src_tok(src_text, truncation=True, max_length=max_len)
        tgt_ids = tgt_tok(tgt_text, truncation=True, max_length=max_len)

        src_tensor = torch.tensor(src_ids, device=device)
        tgt_tensor = torch.tensor(tgt_ids, device=device)

        # Fill up with padding until we reach max_length
        src_padded = pad(src_tensor, (0, max_len - len(src_tensor)), value=PADDING_ID)
        tgt_padded = pad(tgt_tensor)

        src_batch.append(src_padded)
        tgt_batch.append(tgt_padded)
    
    return torch.stack(src_batch), torch.stack(tgt_batch)