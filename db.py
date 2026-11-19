"""Print the saved IMDb Optuna results from a SQLite database."""

import argparse
from pathlib import Path


STUDY_NAME = "imdb_htm_optimization"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database",
        nargs="?",
        type=Path,
        default=Path("imdb_htm_optimization.db"),
        help="Path to the SQLite database (default: %(default)s)",
    )
    args = parser.parse_args()
    database = args.database.expanduser().resolve()
    if not database.is_file():
        parser.error(f"Database file not found: {database}")

    try:
        import optuna
    except ModuleNotFoundError as error:
        if error.name != "optuna":
            raise
        parser.exit(1, "Optuna is required. Run: python3 -m pip install optuna\n")

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{database.as_uri()}?mode=ro&uri=true",
        skip_table_creation=True,
    )
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    except KeyError:
        parser.error(f"Study {STUDY_NAME!r} was not found in {database}")

    trials = study.get_trials(deepcopy=False)
    print(f"Study: {study.study_name}")
    print(f"Direction: {study.direction.name.lower()}")
    print(f"Total trials: {len(trials)}")

    try:
        best = study.best_trial
    except ValueError:
        print("No best trial is available yet.")
    else:
        print(f"Best trial: {best.number}")
        print(f"Best objective value: {best.value}")
        print(f"Best parameters: {best.params}")

    print("\nTrials")
    for trial in trials:
        print(
            f"#{trial.number} {trial.state.name} "
            f"value={trial.value!r} params={trial.params!r}"
        )


if __name__ == "__main__":
    main()
