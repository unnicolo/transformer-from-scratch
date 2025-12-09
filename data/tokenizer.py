# -*- coding: utf-8 -*-

"""
tokenizer.py: # Handle loading SpaCy tokenizers and basic text → token list conversion.
"""

### IMPORTS ###
import spacy
import os

from spacy import Language
from typing import Callable, Iterator, Iterable

### CONSTANTS ###
SPACY_DE_PACKAGE_NAME = "de_core_news_sm"
SPACY_EN_PACKAGE_NAME = "en_core_web_sm"

def load_tokenizers() -> tuple[Language, Language]:
    """Load the german core news and english core web tokenizers.
    
    Args:
        No arguments. 

    Returns:
        A tuple containing the germany and english core news packages.
    """
    # Try to load the german core news package
    try:
        spacy_de = spacy.load(SPACY_DE_PACKAGE_NAME)
    except IOError:
        os.system("python -m spacy download " + SPACY_DE_PACKAGE_NAME)
        spacy_de = spacy.load(SPACY_DE_PACKAGE_NAME)
    
    # Try to load the english core news package
    try:
        spacy_en = spacy.load(SPACY_EN_PACKAGE_NAME)
    except IOError:
        os.system("python -m spacy download " + SPACY_EN_PACKAGE_NAME)
        spacy_en = spacy.load(SPACY_EN_PACKAGE_NAME)
    
    return spacy_de, spacy_en

def tokenize(text: str, tokenizer: Language) -> list[str]:
    """Create tokens from the input text.

    Args:
        text: Text to create tokens from.
    
    Returns:
        A list containing the tokenized input text.
    """
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(
        data_iter: Iterable[tuple], 
        tokenize_fn: Callable[[str], list[str]]) -> Iterator[list[str]]:
    """Yields tokenized text from an iterable of record tuples.

    Each element of `data_iter` is expected to be a tuple containing multiple
    text fields (for example, a source sentence and a target sentence). This
    function selects the field at the given index, applies the provided
    tokenization function, and yields the resulting list of tokens.

    Args:
        data_iter: Iterable of tuples, each containing multiple text fields.
        tokenizer: A function that takes a string and returns a list of tokens.
        index: The position within each tuple from which to extract text.

    Yields:
        A list of token strings corresponding to the tokenized text field
        from each record.
    """ 
    for example in data_iter:
        yield tokenize_fn(example)