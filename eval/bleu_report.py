from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize

import pandas as pd

# input test data: the file name could be changed due to your personal file name
test_df = pd.read_csv('./test_df.csv', index_col=False)

# input generated text data from test: the file name could be changed due to your personal file name
test_df_transfered = pd.read_csv('./transferred.csv', index_col=False).astype(str)

texts_original = test_df['text']

texts_transfered = test_df_transfered['result']

def nltk_bleu(texts_original, texts_transfered):
    
    texts_original = [word_tokenize(original.lower().strip()) for original in texts_original]
    texts_transfered = word_tokenize(texts_transfered.lower().strip())

    return sentence_bleu(texts_original, texts_transfered, weights=(1,0,0,0)) *100


sum_transfered = 0
for x, y in zip(texts_original, texts_transfered):
    n = texts_transfered.shape[0]
    sum_transfered += nltk_bleu([x], y)
    
print('self bleu score: ', sum_transfered / n)
