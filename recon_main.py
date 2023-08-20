import random
import torch
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils.config import ModelConfig
from core.recon import Reconstruction, Classifier
from utils.recon_wrapper import reconstruction_train

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
        
    bert = Classifier(deepcopy(language_model), deepcopy(language_model_config), args)
    bert.load_state_dict(torch.load('./model/classifier/classifier.pt', map_location = config.device))  
    bert.eval()
    for name, params in bert.named_parameters():
        params.requires_grad_(False)         
    bert.to(config.device)
    
    ## Model ##
    recon_model = Reconstruction(bert, tokenizer)
    recon_model.to(config.device)
    
    optimizer = optim.AdamW(recon_model.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)


    if args.MODE == 'train':
        reconstruction_train(args, train_df, valid_df, recon_model, optimizer, scheduler, tokenizer)

    else :
        pass

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest="MODE", type=str, required=True, help='run mode: [train]')
    parser.add_argument('--b', dest="batch_size", type=int, default=64, help='batch size to be used (default: 64)')
    parser.add_argument('--e', dest="epochs", type=int, default=5, help='epochs to be used in \'train\' mode (default: 5)')
    parser.add_argument('--c', dest='classes', type=int, default=5, help='number of class used in \'train\' mode (default: 5)')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parse_argument()
    main(args)
    