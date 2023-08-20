from core.layer import MultiHeadAttentionLayer, PositionWiseFeedForwardLayer

import torch.nn as nn
import torch.nn.functional as F

from utils.config import ModelConfig


config = ModelConfig()

    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate):
        super().__init__()
        self.masked_multi_head_attention_layer = MultiHeadAttentionLayer(d_model, n_head, dropout_rate)
        self.masked_norm_layer = nn.LayerNorm(config.d_embed)

        self.multi_head_attention_layer = MultiHeadAttentionLayer(d_model, n_head, dropout_rate)
        self.multi_norm_layer = nn.LayerNorm(config.d_embed)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(dropout_rate)
        self.position_norm_layer = nn.LayerNorm(config.d_embed)

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask, encoder_output, encoder_mask):
        # x's shape: (batch, seq_len, d_style_emb+d_embed)
        # sen_2_emb's shape: (batch, seq_len, d_style_emb)

        masked_out = self.masked_multi_head_attention_layer(query=x, key=x, value=x, mask=mask)  # out's shape: (batch, seq_len, d_style_emb+d_embed)
        # dropout, residual connection and layer norm
        out = self.masked_norm_layer(x + self.dropout(masked_out))

        multi_out = self.multi_head_attention_layer(query=out, key=encoder_output, value=encoder_output, mask=encoder_mask)  # encoder_output = generative_emb 
        out = self.multi_norm_layer(out + self.dropout(multi_out))
        
        position_out = self.position_wise_feed_forward_layer(x=out)
        out = self.position_norm_layer(out + self.dropout(position_out))

        return out

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate, n_layers):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, dropout_rate) for _ in range(n_layers)])
    
    def forward(self, x, mask, encoder_output, encoder_mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask, encoder_output, encoder_mask)
            # out's shape: (batch, seq_len, d_style_emb+d_embed)
        return out