import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn import ensemble,linear_model,metrics,model_selection
import numpy as np


X,y = make_classification(n_samples = 10000,n_features=25)

xfold1,xfold2,yfold1,yfold2 = model_selection.train_test_split(
    X,
    y,
    test_size=0.5,
    stratify=y
)

logreg = linear_model.LogisticRegression()
rf = ensemble.RandomForestClassifier()
xgbc = xgb.XGBClassifier()

logreg.fit(xfold1,yfold1)
rf.fit(xfold1,yfold1)
xgbc.fit(xfold1,yfold1)

pred_logreg= logreg.predict_proba(xfold2)[:,1]
pred_rf= rf.predict_proba(xfold2)[:,1]
pred_xgbc= xgbc.predict_proba(xfold2)[:,1]

avg_pred = (pred_logreg+pred_rf+pred_xgbc)/3

fold2_preds = np.column_stack((
    pred_logreg,
    pred_rf,
    pred_xgbc,
    avg_pred
))

aucs_fold2 = []

for i in range(fold2_preds.shape[1]):
    auc = metrics.roc_auc_score(yfold2, fold2_preds[:,i])
    aucs_fold2.append(auc)

print(f"Fold-2: LR_AUC = {aucs_fold2[0]}")
print(f"Fold-2: RF_AUC = {aucs_fold2[1]}")
print(f"Fold-2: XGB_AUC = {aucs_fold2[2]}")
print(f"Fold-2: AVG_AUC = {aucs_fold2[3]}")

logreg = linear_model.LogisticRegression()
rf = ensemble.RandomForestClassifier()
xgbc = xgb.XGBClassifier()

logreg.fit(xfold2,yfold2)
rf.fit(xfold2,yfold2)
xgbc.fit(xfold2,yfold2)

pred_logreg= logreg.predict_proba(xfold1)[:,1]
pred_rf= rf.predict_proba(xfold1)[:,1]
pred_xgbc= xgbc.predict_proba(xfold1)[:,1]

avg_pred = (pred_logreg+pred_rf+pred_xgbc)/3

fold1_preds = np.column_stack((
    pred_logreg,
    pred_rf,
    pred_xgbc,
    avg_pred
))

aucs_fold1 = []

for i in range(fold1_preds.shape[1]):
    auc = metrics.roc_auc_score(yfold1, fold1_preds[:,i])
    aucs_fold1.append(auc)

print(f"Fold-1: LR_AUC = {aucs_fold1[0]}")
print(f"Fold-1: RF_AUC = {aucs_fold1[1]}")
print(f"Fold-1: XGB_AUC = {aucs_fold1[2]}")
print(f"Fold-1: AVG_AUC = {aucs_fold1[3]}")
