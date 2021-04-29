import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold,StratifiedKFold

if __name__=="__main__":
    df = pd.read_csv("input/train.csv")
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = StratifiedKFold(n_splits=5)
    for fold,(trn_,val_) in enumerate(kf.split(X=df,y=y)):
        df.loc[val_,'kfold'] = fold
    df.to_csv("input/cat_train_folds.csv",index =False)