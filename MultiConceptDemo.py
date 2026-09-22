import warnings

warnings.filterwarnings(
	"ignore",
	message=r"The CUDA compiler succeeded, but said the following:.*deprecated-gpu-targets.*",
	category=UserWarning,
)

from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

# Purpose: 
# The script demonstrates the use of a hierarchical Tsetlin Machine to learn a multi-concept problem. 
# The task is to predict the XOR of the parity of two randomly generated integers, which are represented as one-hot encoded features. 
# Goal: Predict if the two integers have the same parity (both even or both odd) or different parity (one even and one odd).

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=100, type=int)
	parser.add_argument("--number-of-clauses", default=2, type=int)
	parser.add_argument("--number-of-examples", default=10000, type=int) # Number of training and testing examples
	parser.add_argument("--T", default=128, type=int)
	parser.add_argument("--s", default=21.1, type=float)
	parser.add_argument("--number-of-alternatives", default=64, type=int) # Number of alternatives in the second layer of the hierarchy
	parser.add_argument("--number-of-elements", default=2500, type=int) # Number of unique integers
	parser.add_argument("--number-of-copies", default=2, type=int) # Synonym grupper
	parser.add_argument("--noise", default=0.0, type=float)
	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

features = args.number_of_elements*2 # Features are double the number of elements, since we have two inputs

# Function to generate the labels based on the parity
def get_class_name(label):
	if label == 0:
		return "Same parity"
	else:
		return "Different parity"

# Function to generate a combination table for the given number of elements
def get_combination_table(number_of_elements):
	X = np.zeros((number_of_elements**2, number_of_elements*2), dtype=np.uint32)
	Y = np.zeros(number_of_elements**2, dtype=np.uint32)

	pairs = []

	i = 0
	for first in range(number_of_elements):
		for second in range(number_of_elements):
			X[i, first] = 1 # Set the corresponding feature for the first integer
			X[i, number_of_elements + second] = 1 # Set the corresponding feature for the second integer
			Y[i] = np.logical_xor(first % 2, second % 2) # Compute the label as the XOR of the parity of the two integers
			pairs.append((first, second))
			i += 1

	return X, Y, pairs

def print_combination_table(pairs, labels, predictions=None):
	print("Combination Table.")
	for i, (first, second) in enumerate(pairs):
		# Print the features and the expected label
		first_feature = "x[%d]" % first
		second_feature = "x[%d]" % (args.number_of_elements + second)

		expected = labels[i] # Get the expected label for the current pair

		# If predictions are provided, get the predicted label for the current pair
		line = "%2d + %2d -> %-13s (%s ∧ %s) label=%d" % (
			first,
			second,
			get_class_name(expected),
			first_feature,
			second_feature,
			expected
		)

		if predictions is not None:
			predicted = predictions[i]
			marker = "OK" if predicted == expected else "WRONG"
			
			line += " predicted=%d %-13s %s" % (predicted, get_class_name(predicted), marker)
		print(line)



X_train = np.zeros((args.number_of_examples, features), dtype=np.uint32)

# Class 1: Different parity (one even, one odd) Class 0: Same parity (both even or both odd)
Y_train = np.zeros(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
	# Generate two random integers between 0 and number_of_elements - 1
	x = np.random.randint(args.number_of_elements, size=(2))

	# Set the corresponding features to 1 for the two integers
	X_train[i, x[0]] = 1
	# Set the corresponding feature for the second integer in the second half of the feature vector
	X_train[i, args.number_of_elements + x[1]] = 1

	# Compute the label as the XOR of the parity of the two integers
	Y_train[i] = np.logical_xor(x[0] % 2, x[1] % 2)

# Add noise to the labels based on the specified noise level (changing some labels to the opposite class)
Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)

X_test = np.zeros((args.number_of_examples, features), dtype=np.uint32)
Y_test = np.zeros(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
	x = np.random.randint(args.number_of_elements, size=(2))

	X_test[i, x[0]] = 1
	X_test[i, args.number_of_elements + x[1]] = 1

	Y_test[i] = np.logical_xor(x[0] % 2, x[1] % 2)

tm = MultiClassTsetlinMachine(
	args.number_of_clauses,
	args.T,
	args.s,
	number_of_state_bits=8,
	boost_true_positive_feedback=0,
	hierarchy_structure=(
		(tm.AND_GROUP, features), # The first layer is an AND group that takes the input features
		(tm.OR_ALTERNATIVES, args.number_of_alternatives), # The second layer is an OR group that takes the output of the first layer and creates alternatives
		(tm.AND_ALTERNATIVES, args.number_of_copies) # The third layer is an AND group that takes the output of the second layer and creates copies
	),
	append_negated=False
)

X_combinations, Y_combinations, pairs = get_combination_table(args.number_of_elements)
print_combination_table(pairs, Y_combinations)

for e in range(args.epochs):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("\n Epoch #%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (e+1, result, stop_training-start_training, stop_testing-start_testing))

	# Hierarchy; Each layer of the hierarchy is printed with its type and number of clauses
	tm.print_hierarchy()

	# Evaluate the model on all combinations of the two integers and print the results
	combination_predictions = tm.predict(X_combinations)
	combination_accuracy = 100 * (combination_predictions == Y_combinations).mean()

	print_combination_table(pairs, Y_combinations, combination_predictions)
	print("Combination Accuracy: %.2f%%" % combination_accuracy) # All combinations