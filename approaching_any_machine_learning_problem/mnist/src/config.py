from pathlib import Path
HOMEDIR = Path().resolve().parent.joinpath("python_practice").joinpath("approaching_any_machine_learning_problem").joinpath("mnist")
TRAINING_FILE = HOMEDIR.joinpath("input").joinpath("mnist_train_folds.csv")
MODEL_OUTPUT = HOMEDIR.joinpath("models")