import random
import torch
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils.config import ModelConfig
from core.style_attn import Style_Attention, Classifier, BERT
from utils.style_attn_wrapper import style_attn_training, style_attn_test

from copy import deepcopy

config = ModelConfig()

def main(args):
    ## SET SEED ##
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Dataframe ##
    train_df = pd.read_csv('./data/final_multi/am_train_df.csv', encoding='utf-8', index_col=False)    
    valid_df = pd.read_csv('./data/final_multi/am_valid_df.csv', encoding='utf-8', index_col=False)

    ## Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    ## Pre-trained Classifier ##
    language_model = AutoModel.from_pretrained(config.model_name)
    language_model_config = AutoConfig.from_pretrained(config.model_name)
    
    classifier = Classifier(deepcopy(language_model), deepcopy(language_model_config), args)
    bert = BERT(deepcopy(language_model), deepcopy(language_model_config))
    
    ## Model ##
    model = Style_Attention(tokenizer, bert, classifier, args)
    
    style_params, dec_params = model.get_params()
    
    style_optimizer = optim.AdamW(style_params, lr = 1e-4)
    style_scheduler = optim.lr_scheduler.StepLR(style_optimizer, step_size=1, gamma=0.8)
    
    optimizer = optim.AdamW(dec_params, lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    if args.MODE == 'train':
        
        classifier.load_state_dict(torch.load('./model/classifier/classifier.pt', map_location = config.device), strict=False)
        classifier.eval()
        for name, param in classifier.named_parameters():
            param.requires_grad_(False)
        classifier.to(config.device)
        
        bert.load_state_dict(torch.load('./model/classifier/classifier.pt', map_location = config.device), strict=False) 
        bert.eval()
        for name, params in bert.named_parameters():
            params.requires_grad_(False)         
        bert.to(config.device)
        
        model.load_state_dict(torch.load('./model/adv_module/ADV.pt', map_location = config.device), strict=False)
        for name, param in model.named_parameters():
            if name.startswith('adv'):
                param.requires_grad_(False)
            if name.startswith('dec'):
                param.requires_grad_(False)
            if name.startswith('proj'):
                param.requires_grad_(False)
        model = model.to(config.device)
        style_attn_training(args, train_df, valid_df, model, style_optimizer, optimizer, style_scheduler, scheduler, tokenizer)

    elif args.MODE == 'test':
        
        if args.data == 'Amazon':
            model.load_state_dict(torch.load('./model/CAT/Amazon/CAT.pt', map_location = config.device), strict=False)
            model = model.to(config.device)
            
        elif args.data == 'YELP':
            model.load_state_dict(torch.load('./model/CAT/YELP/CAT.pt', map_location = config.device), strict=False)
            model = model.to(config.device)
        
        print('------------------------- Input Sentence  -------------------------')
        input_sen = input('Input Sentence: ')
        
        input_tokens = tokenizer.encode_plus(
            text = input_sen,
            add_special_tokens = True,
            max_length = config.seq_len,
            padding = 'max_length',
            truncation = True
        )

        sentence = torch.tensor(input_tokens['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(input_tokens['attention_mask'], dtype=torch.float).unsqueeze(0).to(config.device)

        sen_label = []
        source_label = list(map(str, input('Target Style: ').split()))
        if source_label == ['1']:
            sen_label = torch.tensor([0], dtype=torch.long).to(config.device)
        elif source_label == ['2']:
            sen_label = torch.tensor([1], dtype=torch.long).to(config.device)
        elif source_label == ['3']:
            sen_label = torch.tensor([2], dtype=torch.long).to(config.device)
        elif source_label == ['4']:
            sen_label = torch.tensor([3], dtype=torch.long).to(config.device)
        elif source_label == ['5']:
            sen_label = torch.tensor([4], dtype=torch.long).to(config.device)     

        sentence, cls_sigmoid_adv, cls_sigmoid_enc, cls_sigmoid_sty  = style_attn_test(model, sentence, sen_label, attention_mask, tokenizer)
        
        pred_adv = torch.round(cls_sigmoid_adv)
        pred_enc = torch.round(cls_sigmoid_enc)
        pred_sty = torch.round(cls_sigmoid_sty)
        
        print('Generated Sentence: ', sentence)
        print('\n')
        print('LM_out: ', pred_enc)
        print('LM_out_logit: ', cls_sigmoid_enc)
        print('ADV_out: ', pred_adv)
        print('ADV_out_logit: ', cls_sigmoid_adv)
        print('SA_out: ', pred_sty)
        print('SA_out_logit: ', cls_sigmoid_sty)
        
        print('\n')
        print('------------------------------ Complete! ------------------------------')

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest="MODE", type=str, required=True, help='run mode: [train|test]')
    parser.add_argument('--d', dest="data", type=str, help='test data: [Amazon|YELP]')
    parser.add_argument('--b', dest="batch_size", type=int, default=64, help='batch size to be used (default: 64)')
    parser.add_argument('--e', dest="epochs", type=int, default=5, help='epochs to be used in \'train\' mode (default: 5)')
    parser.add_argument('--c', dest='classes', type=int, default=5, help='number of class used in \'train\' mode (default: 5)')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parse_argument()
    main(args)