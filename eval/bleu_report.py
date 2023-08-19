from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize

import pandas as pd

# input original sentence
texts_original = str(['This place is so good I like this place'])

# input transferred sentence
texts_transferred = str(['This place is so good I like this place'])

def nltk_bleu(texts_original, texts_transferred):
    
    texts_original = [word_tokenize(original.lower().strip()) for original in texts_original]
    texts_transferred = word_tokenize(texts_transferred.lower().strip())

    return sentence_bleu(texts_original, texts_transferred, weights=(1,0,0,0)) *100

print(f'self bleu score: {nltk_bleu(texts_original, texts_transferred)}')
