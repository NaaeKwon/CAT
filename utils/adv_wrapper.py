from tqdm import tqdm
from torch.utils.data import DataLoader
from core.dataset import CAT_Dataset 
from utils.config import ModelConfig

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

config = ModelConfig()

############################################# Loss Function ##########################################################


def get_transfer_loss(output, target):
    """
        Input:
            output : [batch, seq_len, vocab_size]   # argmax value = vocab_out
            target : [batch, seq_len]               # vocab token = dec_target
        Returns:
            loss : [loss(tensor)]
    """
    output = output.contiguous().view(-1, output.shape[2])   # output: (batch*seq_len, vocab_size)
    target = target.contiguous().view(-1)                    # target: (batch*seq_len)

    criterion = nn.CrossEntropyLoss(reduction='none')
    transfer_loss = criterion(output, target)
    return transfer_loss  # transfer_loss: (batch*seq_len)


def get_cls_loss(output, target):
    """
        Input: 
            output : [batch, 5]
            target : [batch, 5]
        Returns:
            loss: [loss(tensor)]
    """
    criterion = nn.BCEWithLogitsLoss()
    cls_loss = criterion(output, target)
    return cls_loss


############################################ Plot Loss #########################################################


def recon_loss_plot(args, train, valid, save_path):
    ax = plt.figure().gca()
    ax.plot(train, 'ro-')
    ax.plot(valid, 'go-')
    
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    
    plt.xlim(-1, args.epochs+1)
    
    plt.legend(["Train", "Valid"])
    plt.title('Adversarial Loss')

    plt.savefig(save_path)
    
    
############################################ Training #########################################################


def adv_train(args, train_df, model, style_optimizer, optimizer, tokenizer):

    train_dataset = CAT_Dataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model.train()
    
    epoch_recon_loss = 0
    epoch_cls_loss = 0

    for batch in tqdm(train_loader):
        sentence = batch['sentence'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        label_adv = batch['labels_5_adv'].to(config.device)

        dec_target, vocab_out, logits = model(sentence, attention_mask)

        attention_mask = attention_mask[:, 1:]
        
        pad_mask = (dec_target != 0).float()   # (batch, seq_len)   # 1 = roberta pad idx  # 0 = bert pad idx 
        pad_mask_dim = pad_mask.contiguous().view(-1)   # (batch * seq_len)
        
        ## loss 1 ##
        recon_loss = get_transfer_loss(vocab_out, dec_target)   # (batch * seq_len)
        masked_recon_loss = recon_loss * pad_mask_dim   # (batch * seq_len)
        recon_loss = torch.mean(masked_recon_loss)   # ()
        
        ## loss 2 ##
        cls_loss = get_cls_loss(logits, label_adv)
        
        total_loss = 0.1*recon_loss + cls_loss
        
        style_optimizer.zero_grad()
        optimizer.zero_grad()
        
        total_loss.backward()
        
        style_optimizer.step()
        optimizer.step()
        
        epoch_recon_loss += recon_loss.detach()
        epoch_cls_loss += cls_loss.detach()

    epoch_recon_loss = epoch_recon_loss / len(train_loader)
    epoch_cls_loss = epoch_cls_loss / len(train_loader)

    return epoch_recon_loss, epoch_cls_loss


def adv_evaluate(args, valid_df, model, tokenizer):
    
    valid_dataset = CAT_Dataset(valid_df, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model.eval()

    val_recon_loss = 0
    val_cls_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_loader):
            sentence = batch['sentence'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            label_adv = batch['labels_5_adv'].to(config.device)

            dec_target, vocab_out, logits = model(sentence, attention_mask)

            attention_mask = attention_mask[:, 1:]
            
            pad_mask = (dec_target != 0).float()   # (batch, seq_len)   # 1 = roberta pad idx  # 0 = bert pad idx 
            pad_mask_dim = pad_mask.contiguous().view(-1)   # (batch * seq_len)
            
            ## loss 1 ##
            recon_loss = get_transfer_loss(vocab_out, dec_target)   # (batch * seq_len)
            masked_recon_loss = recon_loss * pad_mask_dim   # (batch * seq_len)
            recon_loss = torch.mean(masked_recon_loss)   # ()
            
            ## loss 2 ##
            cls_loss = get_cls_loss(logits, label_adv)
            
            val_recon_loss += recon_loss.detach()
            val_cls_loss += cls_loss.detach()

        val_recon_loss = val_recon_loss / len(valid_loader)
        val_cls_loss = val_cls_loss / len(valid_loader)

        return val_recon_loss, val_cls_loss


def adversarial_train(args, train_df, valid_df, model, style_optimizer, optimizer, style_scheduler, scheduler, tokenizer):
    
    def createDirectory(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create the directory.")
        
    loss_plot = input("Adversarial Loss File Name: ")
    createDirectory('./model/adv_module/')
    loss_plot_path = './model/adv_module/adv_loss_{}.png'.format(loss_plot)

    print("---- Training Start ----")
    
    train_loss = []
    valid_loss = []

    for epoch in range(args.epochs):
        print("Epoch: {} --------------------------------------------------------------------------------------".format(epoch+1))
        train_recon_loss, train_cls_loss = adv_train(args, train_df, model, style_optimizer, optimizer, tokenizer)
        train_total_loss = train_recon_loss + train_cls_loss
        
        val_recon_loss, val_cls_loss = adv_evaluate(args, valid_df, model, tokenizer)
        val_total_loss = val_recon_loss + val_cls_loss
        
        torch.save(model.state_dict(), './model/adv_module/ADV.pt')
        
        train_loss.append(train_total_loss.detach().cpu().numpy())
        valid_loss.append(val_total_loss.detach().cpu().numpy())
        
        recon_loss_plot(args, train_loss, valid_loss, loss_plot_path)
        
        style_scheduler.step()
        scheduler.step()
    
    print("---- Training Complete ----")