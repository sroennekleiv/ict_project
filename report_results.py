"""Retrain saved IMDB Optuna configurations and report classification metrics."""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

VALIDATION_FRACTION = 0.2
VALIDATION_SEED = 42
STUDY_NAME = "imdb_htm_optimization"
STUDY_DATABASE = "imdb_htm_optimization.db"
MAX_NGRAM = 2
NUM_WORDS = 5000
INDEX_FROM = 2

PARAMETER_TYPES = {
    "T": int,
    "s": float,
    "literal_budget": int,
    "number_of_alternatives": int,
}
LABELS = [0, 1]
CLASS_NAMES = ["negative", "positive"]


def load_completed_trials(database, study_name):
    database_uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        rows = connection.execute(
            """
            SELECT t.number, v.value, p.param_name, p.param_value
            FROM studies AS s
            JOIN trials AS t ON t.study_id = s.study_id
            JOIN trial_values AS v ON v.trial_id = t.trial_id
            JOIN trial_params AS p ON p.trial_id = t.trial_id
            WHERE s.study_name = ? AND t.state = 'COMPLETE'
                AND v.objective = 0 AND v.value_type = 'FINITE'
            ORDER BY v.value DESC, t.number ASC
            """,
            (study_name,),
        ).fetchall()

    trials = {}
    for number, score, name, value in rows:
        trial = trials.setdefault(number, (score, {}))
        if name in PARAMETER_TYPES:
            trial[1][name] = PARAMETER_TYPES[name](value)
    return {number: trial for number, trial in trials.items() if trial[1].keys() == PARAMETER_TYPES.keys()}


# Match the IMDB feature encoding and hierarchy in ImdbHTM.py.
def review_ngrams(review, id_to_word):
    words = [id_to_word[word_id] for word_id in review]
    for n in range(1, MAX_NGRAM + 1):
        for i in range(len(words) - n + 1):
            yield " ".join(words[i:i + n])


def prepare_data(args):
    import keras
    import numpy as np
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split

    imdb = keras.datasets.imdb
    print("Downloading dataset...")
    (train_x, train_y), (eval_x, eval_y) = imdb.load_data(
        num_words=NUM_WORDS, index_from=INDEX_FROM,
    )
    if args.split == "validation":
        train_x, eval_x, train_y, eval_y = train_test_split(
            train_x, train_y, test_size=VALIDATION_FRACTION,
            random_state=VALIDATION_SEED, stratify=train_y,
        )

    word_to_id = {word: index + INDEX_FROM for word, index in imdb.get_word_index().items()}
    word_to_id.update({"<PAD>": 0, "<START>": 1, "<UNK>": 2})
    id_to_word = {index: word for word, index in word_to_id.items()}

    print("Producing bit representation...")
    vocabulary = {}
    for review in train_x:
        for phrase in review_ngrams(review, id_to_word):
            vocabulary[phrase] = vocabulary.get(phrase, 0) + 1
    phrases = [phrase for phrase, count in vocabulary.items() if count >= 10]
    phrase_bit_nr = {phrase: index for index, phrase in enumerate(phrases)}

    def encode_reviews(reviews):
        features = np.zeros((len(reviews), len(phrases)), dtype=np.uint32)
        for i, review in enumerate(reviews):
            for phrase in review_ngrams(review, id_to_word):
                bit = phrase_bit_nr.get(phrase)
                if bit is not None:
                    features[i, bit] = 1
        return features

    X_train = encode_reviews(train_x)
    X_eval = encode_reviews(eval_x)
    Y_train = np.asarray(train_y, dtype=np.uint32)
    Y_eval = np.asarray(eval_y, dtype=np.uint32)

    print("Selecting features...")
    selector = SelectKBest(chi2, k=args.number_of_elements * 2)
    selector.fit(X_train, Y_train)
    X_train = selector.transform(X_train)
    X_eval = selector.transform(X_eval)
    return X_train, Y_train, X_eval, Y_eval


def create_model(args, *, append_negated=True):
    from PyHierarchicalTsetlinMachineCUDA import tm as htm

    return htm.MultiClassTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits=8,
        literal_budget=args.literal_budget,
        boost_true_positive_feedback=0,
        hierarchy_structure=(
            (htm.AND_GROUP, args.number_of_elements * 2),
            (htm.OR_ALTERNATIVES, args.number_of_alternatives),
            (htm.AND_ALTERNATIVES, args.number_of_copies),
        ),
        append_negated=append_negated,
    )


def evaluate(args, params, data):
    from sklearn.metrics import classification_report, confusion_matrix

    X_train, Y_train, X_eval, Y_eval = data
    model_args = SimpleNamespace(**vars(args), **params)
    model = create_model(model_args, append_negated=not args.no_negation)
    try:
        for _ in range(args.epochs - 1):
            model.fit(X_train, Y_train)
            if args.split == "validation":
                model.predict(X_eval)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_eval)
        report = classification_report(
            Y_eval, predictions, labels=LABELS, target_names=CLASS_NAMES,
            digits=4, zero_division=0,
        )
        matrix = confusion_matrix(Y_eval, predictions, labels=LABELS)
        return report, matrix
    finally:
        del model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=2, type=int)
    parser.add_argument("--number-of-elements", default=2500, type=int)
    parser.add_argument("--number-of-copies", default=1, type=int)
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name(STUDY_DATABASE))
    parser.add_argument("--study-name", default=STUDY_NAME)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trials", type=int, nargs="+", metavar="ID", help="Optuna trial numbers")
    selection.add_argument("--top", type=int, default=3, metavar="N",
                           help="Choose the N highest stored validation scores (default: 3)")
    selection.add_argument("--list-trials", action="store_true",
                           help="List completed configurations without training")
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Use the search validation split or refit on all training data for the IMDB test set")
    parser.add_argument("--no-negation", action="store_true",
                        help="Disable appended negated features to match a historical run")
    args = parser.parse_args(argv)
    if min(args.epochs, args.number_of_clauses, args.number_of_elements, args.number_of_copies, args.top) < 1:
        parser.error("Training settings and --top must be positive.")

    try:
        trials = load_completed_trials(args.database, args.study_name)
    except (OSError, sqlite3.Error) as exc:
        parser.error(f"Cannot read Optuna study from {args.database}: {exc}")
    if not trials:
        parser.error(f"No completed IMDB configurations found in study {args.study_name!r}.")

    if args.list_trials:
        for number, (score, params) in trials.items():
            print(f"Trial {number:>3} | stored validation accuracy {score:6.2f}% | {params}")
        return

    selected = list(dict.fromkeys(args.trials)) if args.trials else list(trials)[:args.top]
    missing = [str(number) for number in selected if number not in trials]
    if missing:
        parser.error(f"Trial numbers without a completed IMDB configuration: {', '.join(missing)}")

    print(f"Study {args.study_name!r} | evaluating trials {', '.join(map(str, selected))}", flush=True)
    print(f"Split {args.split} | epochs {args.epochs} | clauses {args.number_of_clauses} | "
          f"features {2 * args.number_of_elements} | copies {args.number_of_copies} | "
          f"appended negated features {not args.no_negation}", flush=True)
    data = prepare_data(args)
    for number in selected:
        score, params = trials[number]
        print(f"\nTrial {number} | stored validation accuracy {score:.2f}% | {params}", flush=True)
        report, matrix = evaluate(args, params, data)
        print(f"Retrained {args.split} classification report\n{report}")
        print("Confusion matrix (rows: actual, columns: predicted)")
        print(f"{'':>10} {'negative':>8} {'positive':>8}")
        for name, (predicted_negative, predicted_positive) in zip(CLASS_NAMES, matrix):
            print(f"{name:>10} {predicted_negative:8d} {predicted_positive:8d}", flush=True)


if __name__ == "__main__":
    main()
