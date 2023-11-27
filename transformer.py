import torch
from torch import nn
from torch.nn import functional as F
import math

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, p_drop):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # calculate numerator
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # calculate denominator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [batch_size, seq_len, d_model]
        '''
        x = x + self.pe[:, : x.size(1)]
        return x
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_drop):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, query, key, value, mask):
        '''
        query: [batch_size, n_heads, q_len, d_k]
        key: [batch_size, n_heads, k_len, d_k]
        value: [batch_size, n_heads, v_len, d_v]
        mask: [batch_size, 1, 1, seq_len]
        '''

        # matmul and scale
        d_k = key.size(-1)
        scale_factor = 1.0/math.sqrt(d_k)
        attention_weights = torch.matmul(query, key.transpose(-1,-2)) * scale_factor
        # mask 
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        # softmax
        attention_weights = self.softmax(attention_weights)
        # dropout
        attention_weights = self.dropout(attention_weights)
        # matmul
        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, p_drop):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = d_model//n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(p_drop)

    def forward(self, query, key, value, mask):
        '''
        query: [batch_size, q_len, d_model]
        key: [batch_size, k_len, d_model]
        value: [batch_size, k_len, d_model]
        mask (source_mask): [batch_size, 1, source_seq_len]
        mask (target_mask): [batch_size, target_seq_len, target_seq_len]
        '''
        batch_size = query.size(0)

        # create linear projections
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # split into n_heads attention heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) # [batch_size, n_heads, q_len, d_k]
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) # [batch_size, n_heads, k_len, d_k]
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2) # [batch_size, n_heads, v_len, d_v]

        # prepare mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        # perform attention
        attention_output, attention_weights = self.attention(Q, K, V, mask=mask)
        
        # concatenate attention heads
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        attention_output = self.W_O(attention_output)

        return attention_output, attention_weights

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, p_drop):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(p_drop)

        self.self_attention = MultiHeadAttention(n_heads, d_model, p_drop)
        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, p_drop)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        '''
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, seq_len]
        '''
        # self attention layer
        attention_output, _ = self.self_attention(x,x,x,mask)
        x = x + self.dropout(attention_output)
        x = self.self_attention_layer_norm(x)

        # feed forward layer
        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.feed_forward_layer_norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_layers, d_model, d_ff, max_len, p_drop):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p_drop)

        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, p_drop) for _ in range(n_layers)])
    
    def forward(self, x, mask):
        '''
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, seq_len]
        '''
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, p_drop):
        super(DecoderLayer, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        
        self.self_attention = MultiHeadAttention(n_heads, d_model, p_drop)
        self.self_attention_layer_norm = nn.LayerNorm(d_model)

        self.encoder_decoder_attention = MultiHeadAttention(n_heads, d_model, p_drop)
        self.encoder_decoder_attention_layer_norm = nn.LayerNorm(d_model)

        self.feed_forward = FeedForwardNetwork(d_model, d_ff, p_drop)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        '''
        x: [batch_size, target_seq_len, d_model]
        encoder_output: [batch_size, source_seq_len, d_model]
        self_mask: [batch_size, target_seq_len, target_seq_len]
        encoder_mask: [batch_size, 1, source_seq_len]
        '''
        # self attention layer 
        attention_output, _ = self.self_attention(x,x,x,self_mask)
        x = x + self.dropout(attention_output)
        x = self.self_attention_layer_norm(x)

        # encoder-decoder layer
        encoder_attention_output, _ = self.encoder_decoder_attention(x,encoder_output,encoder_output,encoder_mask)
        x = x + self.dropout(encoder_attention_output)
        x = self.encoder_decoder_attention_layer_norm(x)
        
        # feed forward layer
        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.feed_forward_layer_norm(x)

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_layers, d_model, d_ff, max_len, p_drop):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p_drop)

        self.decoder_layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, p_drop) for _ in range(n_layers)])

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        '''
        x: [batch_size, target_seq_len, d_model]
        encoder_output: [batch_size, source_seq_len, d_model]
        self_mask: [batch_size, target_seq_len, target_seq_len]
        encoder_mask: [batch_size, 1, source_seq_len]
        '''
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.dropout(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, self_mask, encoder_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_heads, n_layers, d_model, d_ff, pad_tok, max_len=5000, p_drop=0.1):
        super(Transformer, self).__init__()
        self.pad_tok = pad_tok
        self.encoder = Encoder(vocab_size, n_heads, n_layers, d_model, d_ff, max_len, p_drop)
        self.decoder = Decoder(vocab_size, n_heads, n_layers, d_model, d_ff, max_len, p_drop)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def _create_pad_mask(self, x):
        return (x != self.pad_tok).unsqueeze(-2)

    def _create_subsequent_mask(self, x):
        seq_len = x.size(1)
        attention_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

    def forward(self, source, target):
        '''
        source: [batch_size, source_seq_len]
        target: [batch_size, target_seq_len]
        '''
        source_mask = self._create_pad_mask(source)
        target_mask = self._create_pad_mask(target) & self._create_subsequent_mask(target)
        encoder_output = self.encoder(source, source_mask)
        decoder_output = self.decoder(target, encoder_output, target_mask, source_mask)
        output = self.output_projection(decoder_output)

        return output
