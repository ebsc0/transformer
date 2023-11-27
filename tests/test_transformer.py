import torch
from torch import nn, testing
from torch.nn import functional as F

from transformer import *

def test_PositionalEncoding():
    raise NotImplementedError

def test_ScaledDotProductAttention():
    Q = torch.rand(32, 8, 128, 64, dtype=torch.float32)
    K = torch.rand(32, 8, 128, 64, dtype=torch.float32)
    V = torch.rand(32, 8, 128, 64, dtype=torch.float32)

    source = ScaledDotProductAttention(0.0)
    actual, _ = source(Q,K,V,mask=None)
    expected = F.scaled_dot_product_attention(Q,K,V) 

    testing.assert_close(actual, expected)

def test_MultiHeadAttention():
    Q = torch.rand(1, 4, 64, dtype=torch.float32)
    K = torch.rand(1, 4, 64, dtype=torch.float32)
    V = torch.rand(1, 4, 64, dtype=torch.float32)

    source = MultiHeadAttention(8, 64, 0.0)
    actual, _ = source(Q,K,V,mask=None)

    reference = nn.MultiheadAttention(64, 8, bias=False, batch_first=True)
    expected, _ = reference(Q,K,V)

    testing.assert_close(actual, expected) # will evaluate to false since the weights of each linear projection is randomly initialized based on a distribution

    raise NotImplementedError

def test_EncoderLayer():
    raise NotImplementedError

def test_DecoderLayer():
    raise NotImplementedError

def test_Encoder():
    raise NotImplementedError

def test_Decoder():
    raise NotImplementedError

def test_Transformer():
    vocab_size = 1000
    d_model = 512
    d_ff = 2048
    n_heads = 8
    n_layers = 6
    pad_idx = 0
    max_len = 512
    p_drop = 0.1

    Transformer(vocab_size, n_heads, n_layers, d_model, d_ff, pad_idx, max_len, p_drop)