from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize

import pandas as pd

# input original sentence
print('Input original sentence to get bleu score: ')
texts_original = str([input()])

# input transferred sentence
print('Input transferred sentence to get bleu score: ')
texts_transferred = str([input()])

def nltk_bleu(texts_original, texts_transferred):
    
    texts_original = [word_tokenize(original.lower().strip()) for original in texts_original]
    texts_transferred = word_tokenize(texts_transferred.lower().strip())

    return sentence_bleu(texts_original, texts_transferred, weights=(1,0,0,0)) *100

print(f'self bleu score: {nltk_bleu(texts_original, texts_transferred)}')
