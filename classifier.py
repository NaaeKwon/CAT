import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.optim as optim
from tqdm.auto import tqdm
from torchmetrics.classification import MultilabelF1Score
from utils.config import ModelConfig
from typing_extensions import Literal

config = ModelConfig()

## SET SEED ##
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_essay_split(df):
    texts = []
    labels = []

    for i in range(len(df)):
        label = []
        
        texts.append(df['text'].iloc[i])
        
        if df['stars'].iloc[i] == 1.0:
            label = [1.0, 0.0, 0.0, 0.0, 0.0]

        elif df['stars'].iloc[i] == 2.0:
            label = [0.0, 1.0, 0.0, 0.0, 0.0]
            
        elif df['stars'].iloc[i] == 3.0:
            label = [0.0, 0.0, 1.0, 0.0, 0.0]
            
        elif df['stars'].iloc[i] == 4.0:
            label = [0.0, 0.0, 0.0, 1.0, 0.0]
            
        elif df['stars'].iloc[i] == 5.0:
            label = [0.0, 0.0, 0.0, 0.0, 1.0]
        
        labels.append(label)
    
    return texts, labels

class OCEAN_Dataset(Dataset):
    
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        
        self.texts, self.labels = read_essay_split(dataframe)
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        self.text_ids = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens = False,
            max_length = config.seq_len,
            padding = 'max_length',
            truncation = True
        )

        return {
            'ids': torch.tensor(self.text_ids['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.text_ids['attention_mask'], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def binary_accuracy(preds, y):
    rounded_pred = torch.round(torch.sigmoid(preds))
    
    correct = (rounded_pred == y).float()

    acc = torch.mean(correct, dim=0)

    return acc

class Classifier(nn.Module):
    def __init__(self, model, model_config):
        super().__init__()

        self.model = model
        self.model_config = model_config

        classifier_dropout = (model_config.classifier_dropout if model_config.classifier_dropout is not None else model_config.hidden_dropout_prob)

        self.dropout = nn.Dropout(classifier_dropout)

        self.fc1 = nn.Linear(model_config.hidden_size, 256) 
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(256, config.type)

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
    
    
def get_classifier_loss(output, target):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output, target)
    return loss


def get_classifier_f1(output, target):
    """
        Input: 
            output: [batch, 5]   # logit
            target: [batch, 5]   # label
        Return:
            f1_score: [5]    
    """
    metric = MultilabelF1Score(num_labels=5, average=None).to(config.device)
    f1_score = metric(output, target)
    return f1_score


def trainer(classifier, optimizer, train_loader):
    epoch_loss = 0

    epoch_acc_1 = 0
    epoch_acc_2 = 0
    epoch_acc_3 = 0
    epoch_acc_4 = 0
    epoch_acc_5 = 0
    
    epoch_f1_1 = 0
    epoch_f1_2 = 0
    epoch_f1_3 = 0
    epoch_f1_4 = 0
    epoch_f1_5 = 0
    
    classifier.train()    
    for batch in tqdm(train_loader):

        ids = batch['ids'].to(config.device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(config.device, dtype = torch.float)
        label = batch['labels'].to(config.device)
        
        logits_content = classifier(ids, attention_mask)   # (batch, 5)

        loss = get_classifier_loss(logits_content, label)
        accuracy = binary_accuracy(logits_content, label)
        f1_score = get_classifier_f1(torch.sigmoid(logits_content), label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_acc_1 += accuracy[0].item()
        epoch_acc_2 += accuracy[1].item()
        epoch_acc_3 += accuracy[2].item()
        epoch_acc_4 += accuracy[3].item()
        epoch_acc_5 += accuracy[4].item()
        
        epoch_f1_1 += f1_score[0].item()
        epoch_f1_2 += f1_score[1].item()
        epoch_f1_3 += f1_score[2].item()
        epoch_f1_4 += f1_score[3].item()
        epoch_f1_5 += f1_score[4].item()

        epoch_loss += loss.item()
    
    epoch_acc_1 = epoch_acc_1/len(train_loader)
    epoch_acc_2 = epoch_acc_2/len(train_loader)
    epoch_acc_3 = epoch_acc_3/len(train_loader)
    epoch_acc_4 = epoch_acc_4/len(train_loader)
    epoch_acc_5 = epoch_acc_5/len(train_loader)
    
    epoch_f1_1 = epoch_f1_1/len(train_loader)
    epoch_f1_2 = epoch_f1_2/len(train_loader)
    epoch_f1_3 = epoch_f1_3/len(train_loader)
    epoch_f1_4 = epoch_f1_4/len(train_loader)
    epoch_f1_5 = epoch_f1_5/len(train_loader)

    epoch_loss = epoch_loss/len(train_loader)

    return epoch_acc_1, epoch_acc_2, epoch_acc_3, epoch_acc_4, epoch_acc_5, epoch_loss, epoch_f1_1, epoch_f1_2, epoch_f1_3, epoch_f1_4, epoch_f1_5


def validation(classifier, val_loader):
    val_loss = 0

    val_acc_1 = 0
    val_acc_2 = 0
    val_acc_3 = 0
    val_acc_4 = 0
    val_acc_5 = 0
    
    val_f1_1 = 0
    val_f1_2 = 0
    val_f1_3 = 0
    val_f1_4 = 0
    val_f1_5 = 0
    
    classifier.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            ids = batch['ids'].to(config.device, dtype = torch.long)
            attention_mask = batch['attention_mask'].to(config.device, dtype = torch.float)
            label = batch['labels'].to(config.device)
            
            logits_content = classifier(ids, attention_mask)   # (batch, 5)

            loss = get_classifier_loss(logits_content, label)
            accuracy = binary_accuracy(logits_content, label)
            f1_score = get_classifier_f1(torch.sigmoid(logits_content), label)        

            val_acc_1 += accuracy[0].item()
            val_acc_2 += accuracy[1].item()
            val_acc_3 += accuracy[2].item()
            val_acc_4 += accuracy[3].item()
            val_acc_5 += accuracy[4].item()
            
            val_f1_1 += f1_score[0].item()
            val_f1_2 += f1_score[1].item()
            val_f1_3 += f1_score[2].item()
            val_f1_4 += f1_score[3].item()
            val_f1_5 += f1_score[4].item()

            val_loss += loss.item()
        
        val_acc_1 = val_acc_1/len(val_loader)
        val_acc_2 = val_acc_2/len(val_loader)
        val_acc_3 = val_acc_3/len(val_loader)
        val_acc_4 = val_acc_4/len(val_loader)
        val_acc_5 = val_acc_5/len(val_loader)
        
        val_f1_1 = val_f1_1/len(val_loader)
        val_f1_2 = val_f1_2/len(val_loader)
        val_f1_3 = val_f1_3/len(val_loader)
        val_f1_4 = val_f1_4/len(val_loader)
        val_f1_5 = val_f1_5/len(val_loader)

        val_loss = val_loss/len(val_loader)

        return val_acc_1, val_acc_2, val_acc_3, val_acc_4, val_acc_5, val_loss, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5


train_params = {'batch_size': 256,
                'shuffle': True,
                'num_workers': 4
                    }

val_params = {'batch_size': 256,
                'shuffle': True,
                'num_workers': 4
                    }

tokenizer = AutoTokenizer.from_pretrained(config.model_name) 


model = AutoModel.from_pretrained(config.model_name)
model.to(config.device)
model_config = AutoConfig.from_pretrained(config.model_name)

classifier = Classifier(model, model_config)
classifier.to(config.device)

optimizer = optim.AdamW(classifier.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)


train = pd.read_csv('./data/final_multi/y_train_df.csv', encoding='utf-8', index_col=False)
val = pd.read_csv('./data/final_multi/y_valid_df.csv', encoding='utf-8', index_col=False)

train_set = OCEAN_Dataset(train, tokenizer)
val_set = OCEAN_Dataset(val, tokenizer)

train_loader = DataLoader(train_set, **train_params)
val_loader = DataLoader(val_set, **val_params)

epochs = 5
for epoch in range(epochs):
        
    epoch_acc_1, epoch_acc_2, epoch_acc_3, epoch_acc_4, epoch_acc_5, epoch_loss, epoch_f1_1, epoch_f1_2, epoch_f1_3, epoch_f1_4, epoch_f1_5 = trainer(classifier, optimizer, train_loader)
    val_acc_O, val_acc_C, val_acc_E, val_acc_A, val_acc_N, val_loss, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5 = validation(classifier, val_loader)

    torch.save(classifier.state_dict(), './classifier.pt'.format(epoch+1)) 
    
    print("[validation]-----------------------------------------------------------------------------------------------------------------------------------")
    print(f"val_acc_O: {val_acc_O*100:.3f}%, val_acc_C: {val_acc_C*100:.3f}%, val_acc_E: {val_acc_E*100:.3f}%, val_acc_A: {val_acc_A*100:.3f}%, val_acc_N: {val_acc_N*100:.3f}%")
    print(f"val_acc_avg: {(val_acc_O*100 + val_acc_C*100 + val_acc_E*100 + val_acc_A*100 + val_acc_N*100) / 5 :.3f}%")
    print(f"val_f1_1: {val_f1_1:.3f}, val_f1_2: {val_f1_2:.3f}, val_f1_3: {val_f1_3:.3f}, val_f1_4: {val_f1_4:.3f}, val_f1_5: {val_f1_5:.3f}")
    print(f"val_loss: {val_loss:.3f}")

    scheduler.step()