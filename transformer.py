import torch
from torch import nn
from torch.nn import functional as F
import math

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p_drop):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_drop):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, q, k, v, mask):
        # q: [batch_size, n_heads, q_len, d_k]
        # k: [batch_size, n_heads, k_len, d_k]
        # v: [batch_size, n_heads, v_len, d_v]

        # matmul and scale
        d_k = k.size(-1)
        scale_factor = 1.0/math.sqrt(d_k)
        attn = torch.matmul(q, k.transpose(-1,-2)) * scale_factor
        # mask 
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        # softmax
        attn = self.softmax(attn)
        # dropout
        attn = self.dropout(attn)
        # matmul
        v = torch.matmul(attn, v)
        return v, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, p_drop):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = d_model/n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)
        self.attention = ScaledDotProductAttention(p_drop)

    def forward(self, q, k, v, mask):
        # q: [batch_size, q_len, d_model]
        # k: [batch_size, k_len, d_model]
        # v: [batch_size, k_len, d_model]

        # q: [batch_size, n_heads, q_len, d_k]
        q = self.w_q(q).view()
        k = self.w_k(k)
        v = self.w_v(v)

class EncoderLayer(nn.Module):
    pass

class DecoderLayer(nn.Module):
    pass

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p_drop):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(input_dim, hidden_dim, p_drop) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(input_dim, )



class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    pass