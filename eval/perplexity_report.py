from nltk.tokenize import word_tokenize
import pkg_resources
import kenlm
import math
import pandas as pd
import numpy as np

class Evaluator(object):
    
    def __init__(self):
        resource_package = __name__

        # select whether it is amazon or yelp
        ppl_path = 'pre_file/Amazon.binary'
        # ppl_path = 'pre_file/YELP.binary'

        ppl_file = pkg_resources.resource_stream(resource_package, ppl_path)
        
        self.ppl_model = kenlm.Model(ppl_file.name)
        

    def perplexity(self, texts_transfered):
        texts_transfered = [' '.join(word_tokenize(itm.lower().strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):         
            words = line.split()
            num_words = len(words)
            length += num_words
            log_prob = self.ppl_model.score(line)
            sum += log_prob
            
        avg_log_prob = sum/length
        return 10**(-avg_log_prob/np.log(10))
    

def main():
    # input the generated sentence to check perplexity
    print("Input sentence to get perplexity score: ")
    sentence = input()
    
    base = Evaluator()
    ppl = base.perplexity(texts_transfered=[sentence])
    
    print(f" [ppl score]: {ppl:.4f}")
    
if __name__ == '__main__':
    main()
    
