# -*- coding: utf-8 -*-

"""
vocabulary.py: Build an load vocabulary.
"""

### IMPORTS ###

from datasets import load_dataset
from os.path import exists
from spacy import Language, Vocab
from tokenizer import tokenize, yield_tokens
import torch
from torchtext.vocab import build_vocab_from_iterator, Vocab

from tokenizer import load_tokenizers

### CONSTANTS ###
SPECIAL_TOKENS = ["<s>", "</s>", "<blank>", "<unk>"] 
VOCAB_FILENAME = "vocab.pt"
MULTI30K_IDENTIFIER = "WMT/multi30k"

def build_vocabulary(spacy_de: Language, spacy_en: Language) -> tuple[Vocab, Vocab]:
    """Build German and English vocabularies using the HuggingFace Multi30k dataset.
    
    Args:
        spacy_de: A loaded SpaCy language model for German tokenization.
        spacy_en: A loaded SpaCY language model for English tokenization.

    Returns:
        tuple:
            - ``vocab_sry``: TorchText vocabulary for German text.
            - ``vocab_tgt``: TorchText vocabulary for English text.

    """
    print("Loading Multi30k ...")
    
    ds_train = load_dataset("bentrevett/multi30k", split="train")
    ds_val   = load_dataset("bentrevett/multi30k", split="validation")
    ds_test  = load_dataset("bentrevett/multi30k", split="test")


    print("Building German Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(list(ds_train["de"]) + list(ds_val["de"]) + list(ds_test["de"]), 
                     lambda t: tokenize(t, spacy_de)),
        min_freq=2,
        specials=SPECIAL_TOKENS,
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(list(ds_train["en"]) + list(ds_val["en"]) + list(ds_test["en"]), 
                     lambda t: tokenize(t, spacy_en)),
        min_freq=2,
        specials=SPECIAL_TOKENS,
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_de: Language, spacy_en: Language) -> tuple[Vocab, Vocab]:
    """
    Load or build source/target vocabularies for the German–English Transformer.

    Args:
        spacy_de: A loaded SpaCy language model for German tokenization.
        spacy_en: A loaded SpaCy language model for English tokenization.

    Returns:
        tuple:
            vocab_src: The TorchText vocabulary object for German tokens.
            vocab_tgt: The TorchText vocabulary object for English tokens.
    """
    if not exists(VOCAB_FILENAME):
        vocab_src, vocab_tgt = build_vocab_from_iterator(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), VOCAB_FILENAME)
    else:
        vocab_src, vocab_tgt = torch.load(VOCAB_FILENAME)

    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    
    return vocab_src, vocab_tgt

if __name__=='__main__':
   spacy_de, spacy_en = load_tokenizers()
   vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en) 