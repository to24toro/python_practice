import pandas as pd
from itertools import combinations
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb
import copy

def feature_engineering(df,cat_cols):
    combi = list(combinations(cat_cols, 2))
    for c1,c2 in combi:
        df.loc[:,c1+"_"+c2] = df[c1].astype(str)+"_"+df[c2].astype(str)
    return df

def mean_target_encoding(data):
    df = copy.deepcopy(data)
    num_cols = ["fnlwgt","age","capital.gain","capital.loss","hours.per.week"]
    target_mapping = {"<=50K":0,">50K":1}
    df.loc[:,"income"] = df.income.map(target_mapping)
    features = [f for f in df.columns if f not in ["income","kfold"]]
    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna("None")
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:,col] = lbl.transform(df[col])
    encoded_dfs= []
    for fold in range(5):
        df_train = df[df.kfold!=fold].reset_index(drop=True)
        df_valid = df[df.kfold==fold].reset_index(drop=True)
        for col in features:
            mapping_dict = dict(
                df_train.groupby(col)["income"].mean()
            )
            df_valid.loc[:,col+"_enc"] = df_valid[col].map(mapping_dict)
        encoded_dfs.append(df_valid)
    encoded_df = pd.concat(encoded_dfs,axis=0)
    return encoded_df

def run(df,fold):
    features = [f for f in df.columns if f not in ["income","kfold"]]
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth = 7,
        )
    model.fit(x_train,df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(auc)


if __name__=="__main__":
    df = pd.read_csv("input/adult_folds.csv")
    df = mean_target_encoding(df)
    for fold_ in range(5):
        run(df,fold_)
