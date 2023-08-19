from tqdm import tqdm
from torch.utils.data import DataLoader
from core.dataset import CAT_Dataset 
from utils.config import ModelConfig

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
    loss_plot = input("Adversarial Loss File Name: ")
    loss_plot_path = './adv_loss_{}.png'.format(loss_plot)

    print("---- Training Start ----")
    
    train_loss = []
    valid_loss = []

    for epoch in range(args.epochs):
        train_recon_loss, train_cls_loss = adv_train(args, train_df, model, style_optimizer, optimizer, tokenizer)
        train_total_loss = train_recon_loss + train_cls_loss
        
        val_recon_loss, val_cls_loss = adv_evaluate(args, valid_df, model, tokenizer)
        val_total_loss = val_recon_loss + val_cls_loss
        
        torch.save(model.state_dict(), './ADV_{}.pt'.format(epoch+1))
        
        train_loss.append(train_total_loss.detach().cpu().numpy())
        valid_loss.append(val_total_loss.detach().cpu().numpy())
        
        recon_loss_plot(args, train_loss, valid_loss, loss_plot_path)
        
        style_scheduler.step()
        scheduler.step()
    
    print("---- Training Complete ----")
    

def adversarial_test(model, sen_sep, attention_mask, tokenizer, classifier, max_len=64):
    with torch.no_grad():
        model.eval()
        
        enc_in = sen_sep.unsqueeze(0).to(config.device)
        
        enc_embed = model.bert_embed(enc_in)
        
        bert_attn_mask = model.bert.get_extended_attention_mask(attention_mask, enc_embed.size(), config.device)
        enc_out = model.bert_encoder(enc_embed, bert_attn_mask)['last_hidden_state']   # (batch, seq_len, d_embed)

        adv_out = model.adv_layer(enc_out)
        
        enc_mask = model.make_enc_mask(enc_in)
        
    target_indexes = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token)]
       
    for i in range(max_len):
        target_tensor = torch.LongTensor(target_indexes).unsqueeze(0).to(config.device)
        # target_tensor: (1, token)
        target_mask = model.make_dec_mask(target_tensor)
        target_embed = model.dec_embed(target_tensor)
        # target_embed: (1, token, 500)

        with torch.no_grad():
            dec_out = model.decoder(target_embed, target_mask, adv_out, enc_mask)
            vocab_out = model.projector(dec_out)

        pred_token = vocab_out.argmax(2)[:, -1].item()

        target_indexes.append(pred_token)

        if pred_token == tokenizer.convert_tokens_to_ids(tokenizer.sep_token):
            break
    
    target_tokens = tokenizer.convert_ids_to_tokens(target_indexes)
    target_strings = tokenizer.convert_tokens_to_string(target_tokens)

    sentence = ''.join(target_strings[:])
    
    input_tokens = tokenizer.encode_plus(
        text = sentence,
        add_special_tokens = False,
        max_length = config.seq_len,
        padding = 'max_length',
        truncation = True
    )
    
    target_tensor = torch.tensor(input_tokens['input_ids'], dtype=torch.long).unsqueeze(0).to(config.device)
 
    logits = model.classifier(enc_in, attention_mask, adv_out, mode='fc')
    cls_sigmoid_adv = torch.softmax(logits, dim=1).squeeze(0)
    
    logits_enc = model.classifier(enc_in, attention_mask, enc_out, mode='fc')
    cls_sigmoid_enc = torch.softmax(logits_enc, dim=1).squeeze(0)
    
    return sentence, cls_sigmoid_adv, cls_sigmoid_enc