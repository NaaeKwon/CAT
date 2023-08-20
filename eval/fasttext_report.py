import fasttext

# case of multi-class
model = fasttext.train_supervised('./pre_file/am_train_ft.txt') # version of amazon
# model = fasttext.train_supervised('./pre_file/renew_ytrain.txt') # version of yelp


print(f"label num: {model.labels}")
print()


# save fasttext model
model.save_model('./pre_file/ft_train.bin')


# check the performance of the trained model
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    

print('Model Performance: ')    
print_results(*model.test('./pre_file/am_train_ft.txt'))
# print_results(*model.test('./pre_file/renew_ytrain.txt'))
print()


print("Input sentence to get fasttext prediction: ")
sentence = input()
print(model.predict(sentence))