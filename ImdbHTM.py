import numpy as np
from time import time
from datetime import datetime

import keras
import optuna

from opt_einsum import parser
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb

from sklearn.model_selection import train_test_split

from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as htm

import argparse

def format_parameter(value):
	text = ("%g" % value) if isinstance(value, float) else str(value)
	return text.replace("-", "m").replace(".", "p")


def make_default_clauses_output_path(args):
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return "imdb_htm_s%s_T%s_alt%s_%s.txt" % (
		format_parameter(args.s),
		format_parameter(args.T),
		format_parameter(args.number_of_alternatives),
		timestamp,
	)

def default_args(**kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", default=10, type=int)
        parser.add_argument("--number-of-clauses", default=2, type=int)
        parser.add_argument("--number-of-examples", default=10000, type=int) # Number of training and testing examples
        parser.add_argument("--T", default=812, type=int)
        parser.add_argument("--s", default=19.74054, type=float)
        parser.add_argument("--literal-budget", default=20, type=int) # Number of literals in the first layer of the hierarchy
        parser.add_argument("--number-of-alternatives", default=617, type=int) # Number of alternatives in the second layer of the hierarchy
        parser.add_argument("--number-of-elements", default=2500, type=int) # Number of unique integers
        parser.add_argument("--number-of-copies", default=1, type=int) # Synonym grupper
        parser.add_argument("--noise", default=0.0, type=float)
        parser.add_argument("--optuna-search", action="store_true")
        parser.add_argument("--trials", default=20, type=int)
        args = parser.parse_args()
        for key, value in kwargs.items():
                if key in args.__dict__:
                        setattr(args, key, value)
        return args

args = default_args()

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES = args.number_of_elements*2 # Features are double the number of elements

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

if args.optuna_search:
    train_x, test_x, train_y, test_y = train_test_split(
        train_x,
        train_y,
        test_size=0.2,
        random_state=42,
        stratify=train_y,
    )

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

#print("Selected words ngrams used by the Tsetlin Machine:")
#or tm_feature_nr, phrase in enumerate(selected_phrases):
 #   print("%3d -> %s" % (tm_feature_nr, phrase))

X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

if args.optuna_search:
        def objective(trial):
                T = trial.suggest_int("T", 100, 4000, log=True)
                s = trial.suggest_float("s", 2.0, 30.0, log=True)
                literal_budget = trial.suggest_int("literal_budget", 2, 50)
                alternatives = trial.suggest_int(
                        "number_of_alternatives", 10, 1000, log=True
                )

                model = MultiClassTsetlinMachine(
                        args.number_of_clauses,
                        T,
                        s,
                        number_of_state_bits=8,
                        literal_budget=literal_budget,
                        boost_true_positive_feedback=0,
                        hierarchy_structure=(
                                (htm.AND_GROUP, FEATURES),
                                (htm.OR_ALTERNATIVES, alternatives),
                                (htm.AND_ALTERNATIVES, args.number_of_copies),
                        ), append_negated=True,
                )

                validation_accuracy = 0.0
                try:
                        for epoch in range(args.epochs):
                                model.fit(X_train, Y_train)
                                validation_accuracy = float(100 * (model.predict(X_test) == Y_test).mean())
                                print(f"Trial {trial.number}, epoch {epoch + 1}: validation accuracy = {validation_accuracy:.2f}%")

                finally:
                        del model
                return validation_accuracy

        study = optuna.create_study(
                study_name="imdb_htm_optimization",
                storage="sqlite:///imdb_htm_optimization.db",
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=args.trials, n_jobs=1)

        print("Best validation accuracy:", study.best_value)
        print("Best parameters:", study.best_params)

else:
        tm = MultiClassTsetlinMachine(
                args.number_of_clauses, # Number of class labels
                args.T,
                args.s,
                number_of_state_bits=8,
                literal_budget=args.literal_budget,
                boost_true_positive_feedback=0,
                hierarchy_structure=(
                        (htm.AND_GROUP, FEATURES), # The first layer is an AND group that takes the input features
                        (htm.OR_ALTERNATIVES, args.number_of_alternatives), # The second layer is an OR group that takes the output of the first layer
                        (htm.AND_ALTERNATIVES, args.number_of_copies) # The third layer is an AND group that takes the output of the second layer and creates copies
                ),
                append_negated=True
        )

        # Store results (including the rules of the clauses) for each epoch
        results = {}

        print(f"\nAccuracy over {args.epochs} epochs:\n")
        for e in range(args.epochs):
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()

                start_testing = time()
                result = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))

                # Print hierarchy structure with the names of the selected phrases
                tm.print_hierarchy(feature_names=selected_phrases)

        print("----------------------------------------------------------------------------------------------")