import torch

class ModelConfig:
    """
    Model configuration
    """
    def __init__(self):
        self.seq_len = 128
        self.d_embed = 512  
        self.d_model = 512

        self.d_ff = 1024

        self.n_head = 4
        self.n_layer = 6

        self.d_style = 64
        
        self.type = 5
        self.dropout_rate = 0.15
        
        self.model_name = 'prajjwal1/bert-medium'   # pre-trained LM model name
        self.vocab_size = 30522                     # pre-trained LM's vacabulary size
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")