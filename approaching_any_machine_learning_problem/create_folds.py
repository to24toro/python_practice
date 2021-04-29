import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
HOMEDIR = Path().resolve().parent.joinpath("approaching_any_machine_learning_problem")

if __name__=="__main__":
    df = pd.read_csv(HOMEDIR.joinpath("input").joinpath("adult.csv"))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = KFold(n_splits=5)
    for fold,(trn_,val_) in enumerate(kf.split(X=df)):
        df.loc[val_,'kfold'] = fold
    df.to_csv(HOMEDIR.joinpath("input").joinpath("adult_folds.csv"),index =False)