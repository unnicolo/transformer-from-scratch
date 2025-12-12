# -*- coding: utf-8 -*-

"""
tokenizer.py: Handle loading HuggingFace pretrained tokenizers.
"""

### IMPORTS ###
from transformers import AutoTokenizer, PreTrainedTokenizer

### CONSTANTS ###
BERT_BASE_GERMAN_CASED = "bert-base-german-cased"
BERT_BASE_ENGLISH_CASED = "bert-base-cased"

def load_tokenizers() -> tuple[PreTrainedTokenizer, PreTrainedTokenizer]:
    """Load pretrained BERT German and English cased tokenizers.
    
    Args:
        No arguments. 

    Returns:
        A tuple containing BERT pretrained German and English cased tokenizers.
    """
    src_tok = AutoTokenizer.from_pretrained(BERT_BASE_GERMAN_CASED)
    tgt_tok = AutoTokenizer.from_pretrained(BERT_BASE_ENGLISH_CASED)

    return src_tok, tgt_tok