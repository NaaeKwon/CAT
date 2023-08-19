import torch
from torch.utils.data import Dataset
from utils.config import ModelConfig

config = ModelConfig()


class Dataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = read_split(dataframe)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        self.text_tokens = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens = True,
            max_length = config.seq_len+1,
            padding = 'max_length',
            truncation = True
        )
       
        return {
                'sentence' : torch.tensor(self.text_tokens['input_ids'], dtype=torch.long),
                'attention_mask' : torch.tensor(self.text_tokens['attention_mask'], dtype=torch.float)
                }

def read_split(df):
    texts = []

    for i in range(len(df)):
        
        texts.append(df['text'].iloc[i])
    
    return texts


class CAT_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts, self.labels, self.labels_5, self.labels_5_adv = read_split_trans(dataframe)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        self.text_tokens = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens = True,
            max_length = config.seq_len+1,
            padding = 'max_length',
            truncation = True
        )
       
        return {
                'sentence' : torch.tensor(self.text_tokens['input_ids'], dtype=torch.long),
                'attention_mask' : torch.tensor(self.text_tokens['attention_mask'], dtype=torch.float),
                'labels' : torch.tensor(self.labels[idx], dtype=torch.long),
                'labels_5': torch.tensor(self.labels_5[idx], dtype=torch.float),
                'labels_5_adv': torch.tensor(self.labels_5_adv[idx], dtype=torch.float)
                }

def read_split_trans(df):
    texts = []
    labels = []
    labels_5 = []
    labels_5_adv = []

    for i in range(len(df)):
        
        label = []
        label_5 = []
        label_5_adv = []
        
        texts.append(df['text'].iloc[i])
        
        if df['stars'].iloc[i] == 1:
            label.append(int(0))
            label_5 = [1.0, 0.0, 0.0, 0.0, 0.0]
            label_5_adv = [0.2, 0.2, 0.2, 0.2, 0.2]

        elif df['stars'].iloc[i] == 2:
            label.append(int(1))
            label_5 = [0.0, 1.0, 0.0, 0.0, 0.0]
            label_5_adv = [0.2, 0.2, 0.2, 0.2, 0.2]
            
        elif df['stars'].iloc[i] == 3:
            label.append(int(2))
            label_5 = [0.0, 0.0, 1.0, 0.0, 0.0]
            label_5_adv = [0.2, 0.2, 0.2, 0.2, 0.2]
            
        elif df['stars'].iloc[i] == 4:
            label.append(int(3))
            label_5 = [0.0, 0.0, 0.0, 1.0, 0.0]
            label_5_adv = [0.2, 0.2, 0.2, 0.2, 0.2]
            
        elif df['stars'].iloc[i] == 5:
            label.append(int(4))
            label_5 = [0.0, 0.0, 0.0, 0.0, 1.0]
            label_5_adv = [0.2, 0.2, 0.2, 0.2, 0.2]
            
        labels.append(label)
        labels_5.append(label_5)
        labels_5_adv.append(label_5_adv)
    
    return texts, labels, labels_5, labels_5_adv