import torch
import torch.nn as nn

from model.transformer import Transformer

def make_model(
    src_vocab, 
    tgt_vocab, 
    N=6, 
    d_model=512, 
    d_ff=2048, 
    h=8, 
    dropout=0.1
):
    return Transformer(d_model, d_ff, h, dropout, src_vocab, tgt_vocab)

make_model(11, 11, 2)
