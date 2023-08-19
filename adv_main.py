import random
import torch
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils.config import ModelConfig
from core.adv import Adversarial, Classifier, BERT
from utils.adv_wrapper import adversarial_train, adversarial_test

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
    train_df = pd.read_csv('./train_df.csv', encoding='utf-8', index_col=False)    
    valid_df = pd.read_csv('./valid_df.csv', encoding='utf-8', index_col=False)

    ## Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    ## Pre-trained Classifier ##
    language_model = AutoModel.from_pretrained(config.model_name)
    language_model_config = AutoConfig.from_pretrained(config.model_name)
    
    classifier = Classifier(deepcopy(language_model), deepcopy(language_model_config), args)
    classifier.load_state_dict(torch.load('./classifier.pt', map_location = config.device), strict=False)
    classifier.eval()
    for name, param in classifier.named_parameters():
        param.requires_grad_(False)
    classifier.to(config.device)

    bert = BERT(deepcopy(language_model), deepcopy(language_model_config))
    bert.load_state_dict(torch.load('./classifier.pt', map_location = config.device), strict=False) # 새로 학습한 분류기
    bert.eval()
    for name, params in bert.named_parameters():
        params.requires_grad_(False)         
    bert.to(config.device)
    
    ## Model ##
    model = Adversarial(tokenizer, bert, classifier)
    
    style_params, dec_params = model.get_params()
    
    style_optimizer = optim.AdamW(style_params, lr = 1e-5)
    style_scheduler = optim.lr_scheduler.StepLR(style_optimizer, step_size=1, gamma=0.8)
    
    optimizer = optim.AdamW(dec_params, lr = 1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    if args.MODE == 'train':
        model.load_state_dict(torch.load('./Recon_5.pt', map_location = config.device), strict=False)
        for name, param in model.named_parameters():
            if name.startswith('dec'):
                param.requires_grad_(False)
            if name.startswith('proj'):
                param.requires_grad_(False)
        model = model.to(config.device)
        adversarial_train(args, train_df, valid_df, model, style_optimizer, optimizer, style_scheduler, scheduler, tokenizer)

    elif args.MODE == 'test':
        PATH = './ADV.pt'
        model.load_state_dict(torch.load(PATH, map_location = config.device), strict=False)
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

        sentence, cls_sigmoid_adv, cls_sigmoid_enc  = adversarial_test(model, sentence, attention_mask, tokenizer, classifier)
        
        pred_adv = torch.round(cls_sigmoid_adv)
        pred_enc = torch.round(cls_sigmoid_enc)
        
        print('Generated Sentence: ', sentence)
        print('\n')
        print('LM_out: ', pred_enc)
        print('LM_out_logit: ', cls_sigmoid_enc)
        print('ADV_out: ', pred_adv)
        print('ADV_out_logit: ', cls_sigmoid_adv)
        
        print('\n')
        print('------------------------------ Complete! ------------------------------')

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest="MODE", type=str, required=True, help='run mode: [train|test]')
    parser.add_argument('--b', dest="batch_size", type=int, default=64, help='batch size to be used (default: 64)')
    parser.add_argument('--e', dest="epochs", type=int, default=5, help='epochs to be used in \'train\' mode (default: 5)')
    parser.add_argument('--c', dest='classes', type=int, default=5, help='number of class used in \'train\' mode (default: 5)')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parse_argument()
    main(args)