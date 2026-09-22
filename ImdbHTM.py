import numpy as np
from time import time

import keras

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=10, type=int)
	parser.add_argument("--number-of-clauses", default=2, type=int)
	parser.add_argument("--number-of-examples", default=10000, type=int) # Number of training and testing examples
	parser.add_argument("--T", default=128, type=int)
	parser.add_argument("--s", default=21.1, type=float)
	parser.add_argument("--number-of-alternatives", default=96, type=int) # Number of alternatives in the second layer of the hierarchy
	parser.add_argument("--number-of-elements", default=2500, type=int) # Number of unique integers
	parser.add_argument("--number-of-copies", default=1, type=int) # Synonym grupper
	parser.add_argument("--noise", default=0.0, type=float)
	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES = args.number_of_elements*2 # Features are double the number of elements, since we have two inputs

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
                terms.append(id_to_word[word_id])
        
        for N in range(1,MAX_NGRAM+1):
                grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
                for gram in grams:
                        phrase = " ".join(gram)
                        
                        if phrase in vocabulary:
                                vocabulary[phrase] += 1
                        else:
                                vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
        if vocabulary[phrase] < 10:
                continue

        phrase_bit_nr[phrase] = bit_nr
        bit_nr_phrase[bit_nr] = phrase
        bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
                terms.append(id_to_word[word_id])

        for N in range(1,MAX_NGRAM+1):
                grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
                for gram in grams:
                        phrase = " ".join(gram)
                        if phrase in phrase_bit_nr:
                                X_train[i,phrase_bit_nr[phrase]] = 1

        Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
                terms.append(id_to_word[word_id])

        for N in range(1,MAX_NGRAM+1):
                grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
                for gram in grams:
                        phrase = " ".join(gram)
                        if phrase in phrase_bit_nr:
                                X_test[i,phrase_bit_nr[phrase]] = 1                             

        Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)

selected_phrases = [bit_nr_phrase[int(original_bit)] for original_bit in selected_features]

print("\nSelected words/ngrams used by the Tsetlin Machine:\n")
for tm_feature_nr, phrase in enumerate(selected_phrases):
    print("%3d -> %s" % (tm_feature_nr, phrase))

X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

tm = MultiClassTsetlinMachine(
	args.number_of_clauses,
	args.T,
	args.s,
	number_of_state_bits=8,
	boost_true_positive_feedback=0,
	hierarchy_structure=(
		(tm.AND_GROUP, FEATURES), # The first layer is an AND group that takes the input features
		(tm.OR_ALTERNATIVES, args.number_of_alternatives), # The second layer is an OR group that takes the output of the first layer
		(tm.AND_ALTERNATIVES, args.number_of_copies) # The third layer is an AND group that takes the output of the second layer and creates copies
	),
	append_negated=False
)
print(f"\nAccuracy over {args.epochs} epochs:\n")
for e in range(args.epochs):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
        tm.print_hierarchy()

        # Print hierarchy structure with the names of the selected phrases
        
        print("----------------------------------------------------------------------------------------------")