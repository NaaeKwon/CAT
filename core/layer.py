from torch.autograd import Variable
from utils.config import ModelConfig

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

config = ModelConfig()

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate):
        # qkv_fc_layer's shape: (d_embed, d_model)
        # fc_layer's shape: (d_model, d_embed)
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        self.query_fc_layer = nn.Linear(config.d_embed, config.d_model)
        self.key_fc_layer = nn.Linear(config.d_embed, config.d_model)
        self.value_fc_layer = nn.Linear(config.d_embed, config.d_model)

        self.fc_layer = nn.Linear(config.d_model, config.d_embed)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):  
        # ENCODER: query, key, value's shape: (n_batch, seq_len, d_embed)
        # mask's shape: (n_batch, 1, seq_len, seq_len)
        n_batch = query.shape[0]  # get n_batch

        def transform(x, qkv_fc_layer):  # reshape (n_batch, seq_len, d_embed) -> (n_batch, h, seq_len, d_k)
            out = qkv_fc_layer(x)  # out's shape: (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.n_head, self.d_model//self.n_head)  
            # out's shape: (n_batch, seq_len, h, d_k), d_k = d_model/n_head
            out = out.transpose(1, 2)  # out's shape: (n_batch, h, seq_len, d_k)
            return out

        # query, key, value's shape: (n_batch, h, seq_len, d_k)
        query = transform(query, self.query_fc_layer) 
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        out = self.calculate_attention(query, key, value, mask, self.dropout)  # out's shape: (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2)  # out's shape: (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model)  # out's shape: (n_batch, seq_len, d_model)
        out = self.fc_layer(out)  # out's shape: (n_batch, seq_len, d_embed)
        return out

    def calculate_attention(self, query, key, value, mask, dropout=None):
        # query, key, value's shape: (n_batch, h, seq_len, d_k)
        d_k = key.size(-1)  # get d_k
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # key.transpose result: (n_batch, h, d_k, seq_len)
        # Q x K^T, attention_score's shape: (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)  # scaling

        if mask is not None:  
            attention_score = attention_score.masked_fill(mask==0, -1e9)  # masking (Decoder's Masked Multi-Attention Layer)
        out = F.softmax(attention_score, dim=-1)  # softmax, attention_prob's shape: (n_batch, h, seq_len, seq_len)
        
        if dropout is not None:  
            out = dropout(out) 
        out = torch.matmul(out, value)  
        # attention_prob X value, out's shape: (n_batch, h, seq_len, d_k)
        return out

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.first_fc_layer = nn.Linear(config.d_embed, config.d_ff)
        self.second_fc_layer = nn.Linear(config.d_ff, config.d_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # input's shape: (batch, seq_len, d_embed)
        out = self.first_fc_layer(x)  # out's shape: (batch, seq_len, d_ff)
        out = F.relu(out)  # out's shape: (batch, seq_len, d_ff)
        out = self.dropout(out)  # out's shape: (batch, seq_len, d_ff)
        out = self.second_fc_layer(out)  # out's shape: (batch, seq_len, d_embed)
        return out
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = Embedding(d_embed = config.d_embed, vocab = vocab)
        self.positional_encoding = PositionalEncoding(d_embed = config.d_embed, max_seq_len = config.seq_len)

    def forward(self, x):
        # input x's shape: (batch, seq_len) = noise_inp_token        
        out = self.embedding(x) 
        out = self.positional_encoding(out) 
        # out's shape: (batch, seq_len, d_embed) = sen_1_emb   
        return out

class Embedding(nn.Module):
    def __init__(self, d_embed, vocab):
        super().__init__()
        self.vocab_embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=d_embed)
        self.d_embed = d_embed
    
    def forward(self, x):
        # input x's shape: (batch, seq_len) = noise_inp_token
        out = self.vocab_embedding(x) * math.sqrt(self.d_embed)  
        # self.embedding(x)'s shape: (batch, seq_len, d_embed)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len):
        super().__init__()

        self.register_buffer('encoding', torch.zeros(max_seq_len, d_embed))
        self.register_buffer('position', torch.arange(0, max_seq_len).unsqueeze(1))
        self.register_buffer('div_term', torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0)/d_embed)))
        
        self.encoding[:, 0::2] = torch.sin(self.position * self.div_term)
        self.encoding[:, 1::2] = torch.cos(self.position * self.div_term)
        # encoding's shape: (seq_len, d_embed)

        self.encoding = self.encoding.unsqueeze(0)
        # encoding's shape: (1, seq_len, d_embed)

        # self.encoding = encoding
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # input x's shape: (batch, seq_len, d_embed) = embedding output
        out = x + Variable(self.encoding[:, :x.size(1), :], requires_grad=False)
        # out's shape: (batch, seq_len, d_embed)
        # out = self.dropout(out)
        return out