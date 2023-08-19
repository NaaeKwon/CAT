import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from core.layer import TransformerEmbedding
from core.transformer import Decoder
from utils.config import ModelConfig

config = ModelConfig()

class Reconstruction(nn.Module):
    def __init__(self, classifier, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.vocab
        
        self.classifier = classifier.model
        self.bert_embed = self.classifier.embeddings 
        self.bert_encoder = self.classifier.encoder 
        
        self.dec_embed = TransformerEmbedding(vocab=self.vocab)

        self.decoder = Decoder(d_model=config.d_model, n_head=config.n_head, dropout_rate=config.dropout_rate, n_layers=config.n_layer)

        self.projector = nn.Linear(config.d_embed, config.vocab_size)

    def forward(self, sentence, attention_mask):

        enc_input = sentence[:, 1:]   
        dec_input = sentence[:, :-1]  
        dec_target = sentence[:, 1:]  
        
        attention_mask = attention_mask[:, 1:]
        
        ## Make Encoder Mask ##
        enc_mask = self.make_enc_mask(enc_input)      # (batch, 1, 1, seq_len)

        ## Make Encoder Output ##
        enc_embed = self.bert_embed(enc_input)         # (batch, seq_len, d_embed:512)
        
        bert_attn_mask = self.classifier.get_extended_attention_mask(attention_mask, enc_embed.size(), config.device)
        enc_out = self.bert_encoder(enc_embed, bert_attn_mask)['last_hidden_state']   # (batch, seq_len, d_embed)
        
        ## Make Decoder Mask ##
        dec_mask = self.make_dec_mask(dec_input)       # (batch, 1, seq_len, seq_len)

        ## Make Decoder Output ##
        dec_embed = self.dec_embed(dec_input)                                # (batch, seq_len, d_embed)
        dec_out = self.decoder(dec_embed, dec_mask, enc_out, enc_mask)   # (batch, seq_len, d_embed)

        vocab_out = self.projector(dec_out)    # (batch, seq_len, vocab_size)
        
        return dec_target, vocab_out
    

    def make_enc_mask(self, inp_tokens, pad_idx=0):
        # inp_tokens: (batch, seq_len)
        
        enc_mask = (inp_tokens != pad_idx).unsqueeze(1).unsqueeze(2)
        # enc_mask: (batch, 1, 1, seq_len)

        return enc_mask
    

    def make_dec_mask(self, inp_tokens, pad_idx=0):
        # inp_tokens: (batch, seq_len)

        dec_pad_mask = (inp_tokens != pad_idx).unsqueeze(1).unsqueeze(2)
        # dec_pad_mask: (batch, 1, 1, seq_len)

        seq_len = inp_tokens.shape[1]

        dec_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=config.device)).bool()
        # dec_sub_mask: (seq_len, seq_len)

        dec_mask = dec_pad_mask & dec_sub_mask
        # dec_mask: (batch, 1, seq_len, seq_len)

        return dec_mask
    
    

class Classifier(nn.Module):
    def __init__(self, model, model_config, args):
        super().__init__()

        self.model = model
        self.model_config = model_config

        classifier_dropout = (model_config.classifier_dropout if model_config.classifier_dropout is not None else model_config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        self.fc1 = nn.Linear(model_config.hidden_size, 256) 
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(256, args.classes)

    def forward(self, ids, attention_mask):
        if ids.dim()==2:
            embedding = self.model.embeddings(ids)  # (batch, seq_len, hidden)

        else:
            ids = F.softmax(ids, dim=-1)
            embedding = ids.matmul(self.model.embeddings.word_embeddings.weight) # (batch, seq_len, hidden)
            
        attention_mask = self.model.get_extended_attention_mask(attention_mask, embedding.size(), config.device)
        last_hidden_state = self.model.encoder(embedding, attention_mask)['last_hidden_state']
        last_hidden_state = self.dropout(last_hidden_state)

        context = last_hidden_state.mean(dim=1)   # (batch, hidden_size)

        logits_content = self.fc1(context)
        logits_content = self.relu(logits_content)
        logits_content = self.fc2(logits_content)    

        return logits_content