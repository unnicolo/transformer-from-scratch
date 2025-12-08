# -*- coding: utf-8 -*-

"""
vocabulary.py: Build an load vocabulary.
"""

### IMPORTS ###

from spacy import Language, Vocab
from tokenizer import tokenize, yield_tokens
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator, Vocab

### CONSTANTS ###
SPECIAL_TOKENS = ["<s>", "</s>", "<blank>", "<unk>"] 

def build_vocabulary(spacy_de: Language, spacy_en: Language) -> tuple[Vocab, Vocab]:
    def tokenize_de(text: str) -> list[str]:
        """Tokenizes a German text string using the preloaded German spaCy model.

        This function is a wrapper around `tokenize` that automatically uses
        the German spaCy tokenizer `spacy_de`.

        Args:
            text: The input German string to tokenize.

        Returns:
            A list of token strings extracted from the input text.
        """
        return tokenize(text, spacy_de)
    
    def tokenize_en(text: str) -> list[str]:
        """Tokenizes an English text string using the preloaded English spaCy model.

        This function is a wrapper around `tokenize` that automatically uses
        the English spaCy tokenizer `spacy_en`.

        Args:
            text: The input English string to tokenize.

        Returns:
            A list of token strings extracted from the input text.
        """
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_feq=2,
        specials=SPECIAL_TOKENS,
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=SPECIAL_TOKENS,
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

if __name__=='__main__':
    build_vocabulary