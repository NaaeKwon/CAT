from tqdm import tqdm
from torch.utils.data import DataLoader
from core.dataset import Dataset 
from utils.config import ModelConfig

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

config = ModelConfig()

############################################# Loss Function ##########################################################


def get_recon_loss(output, target):
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
    recon_loss = criterion(output, target)
    return recon_loss  # recon_loss: (batch*seq_len)


############################################ Plot Loss #########################################################


def recon_loss_plot(args, train, valid, save_path):
    ax = plt.figure().gca()
    ax.plot(train, 'ro-')
    ax.plot(valid, 'go-')
    
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    
    plt.xlim(-1, args.epochs+1)
    
    plt.legend(["Train", "Valid"])
    plt.title('Reconstruction Loss')

    plt.savefig(save_path)
    
    
############################################ Reconstruct Training #########################################################


def recon_train(args, train_df, model, optimizer, tokenizer):

    train_dataset = Dataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model.train()

    epoch_recon_loss = 0

    for batch in tqdm(train_loader):
        sentence = batch['sentence'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)

        dec_target, vocab_out = model(sentence, attention_mask)
        
        pad_mask = (dec_target != 0).float()   # (batch, seq_len)    
        pad_mask_dim = pad_mask.contiguous().view(-1)   # (batch * seq_len)
        
        recon_loss = get_recon_loss(vocab_out, dec_target)   # (batch * seq_len)
        masked_recon_loss = recon_loss * pad_mask_dim   # (batch * seq_len)
        reconstruction_loss = torch.mean(masked_recon_loss)   # ()
        
        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()
        
        epoch_recon_loss += reconstruction_loss.detach()

    epoch_recon_loss = epoch_recon_loss / len(train_loader)

    return epoch_recon_loss


def recon_evaluate(args, valid_df, model, tokenizer):
    
    valid_dataset = Dataset(valid_df, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.eval()

    val_recon_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_loader):
            sentence = batch['sentence'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)

            dec_target, vocab_out = model(sentence, attention_mask)
            
            pad_mask = (dec_target != 0).float()
            pad_mask_dim = pad_mask.contiguous().view(-1)
            
            recon_loss = get_recon_loss(vocab_out, dec_target)
            masked_recon_loss = recon_loss * pad_mask_dim
            reconstruction_loss = torch.mean(masked_recon_loss)
            
            val_recon_loss += reconstruction_loss.detach()
            
        val_recon_loss = val_recon_loss / len(valid_loader)

        return val_recon_loss


def reconstruction_train(args, train_df, valid_df, model, optimizer, scheduler, tokenizer):
    loss_plot = input("Reconstruction Loss File Name: ")
    loss_plot_path = './recon_loss_{}.png'.format(loss_plot)

    print("---- Training Start ----")
    
    train_loss = []
    valid_loss = []

    for epoch in range(args.epochs):
        train_recon_loss = recon_train(args, train_df, model, optimizer, tokenizer)
        valid_recon_loss = recon_evaluate(args, valid_df, model, tokenizer)
    
        torch.save(model.state_dict(), './Recon_{}.pt'.format(epoch+1))

        train_loss.append(train_recon_loss.detach().cpu().numpy())
        valid_loss.append(valid_recon_loss.detach().cpu().numpy())
        
        recon_loss_plot(args, train_loss, valid_loss, loss_plot_path)
        
        scheduler.step()
    
    print("---- Training Complete ----")
 
 
def reconstruction_test(recon_model, sen_sep, attention_mask, tokenizer, max_len=512):
    
    with torch.no_grad():
        recon_model.eval()
        
        enc_in = sen_sep.unsqueeze(0).to(config.device)
        attention_mask = attention_mask[:, 1:]
        
        enc_mask = recon_model.make_enc_mask(enc_in)
        enc_embed = recon_model.bert_embed(enc_in)
        
        bert_attn_mask = recon_model.classifier.get_extended_attention_mask(attention_mask, enc_embed.size(), config.device)
        enc_out = recon_model.bert_encoder(enc_embed, bert_attn_mask)['last_hidden_state']   # (batch, seq_len, d_embed)
        # enc_out: (1, seq_len, d_embed)
        
    target_indexes = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token)]
       
    for i in range(max_len):
        target_tensor = torch.LongTensor(target_indexes).unsqueeze(0).to(config.device)
        # target_tensor: (1, token)
        target_mask = recon_model.make_dec_mask(target_tensor)
        target_embed = recon_model.dec_embed(target_tensor)
        # target_embed: (1, token, 500)

        with torch.no_grad():
            dec_out = recon_model.decoder(target_embed, target_mask, enc_out, enc_mask)
            vocab_out = recon_model.projector(dec_out)

        pred_token = vocab_out.argmax(2)[:, -1].item()

        target_indexes.append(pred_token)

        if pred_token == tokenizer.convert_tokens_to_ids(tokenizer.sep_token):
            break

    target_tokens = tokenizer.convert_ids_to_tokens(target_indexes)
    target_strings = tokenizer.convert_tokens_to_string(target_tokens)

    sentence = ''.join(target_strings[:])
    
    return sentence