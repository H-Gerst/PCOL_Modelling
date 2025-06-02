import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import joblib
import json
import pandas as pd

def false_pos_false_neg(cm):
    tn = cm[0,0]
    tp = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]

    false_pos_rate = fp/(fp + tn)
    false_neg_rate = fn/(fn + tp)

    print("false positive rate:", false_pos_rate)
    print("false negative rate:", false_neg_rate)

def main():
    # importing the three models
    loaded_model_uh = joblib.load("best_model_unhashed.joblib")
    loaded_model_hc = joblib.load("best_model_hashed_count.joblib")
    loaded_model_hb = joblib.load("best_model_hashed_binary.joblib")
    print("Models loaded successfully.")

    # Loading files
    holdout_hc = pd.read_csv("validation_ames_data_hc.csv")
    holdout_hb = pd.read_csv("validation_ames_data_hb.csv")
    holdout_uh = pd.read_csv("validation_ames_data_uh.csv")

    # Specifying HASHED + binary holdout data
    x_holdout_hb = holdout_hb.drop(columns=['AMES'])
    y_holdout_hb = holdout_hb['AMES']

    # Specifying HASHED + count holdout data
    x_holdout_hc = holdout_hc.drop(columns=['AMES'])
    y_holdout_hc = holdout_hc['AMES']

    # Specifying UNHASHED holdout data
    x_holdout_uh = holdout_uh.drop(columns=['AMES'])
    y_holdout_uh = holdout_uh['AMES']


    # Hashed + binary validation
    y_pred_hb = loaded_model_hb.predict(x_holdout_hb)
    y_prob_hb = loaded_model_hb.predict_proba(x_holdout_hb)[:,1]

    print("Hashed + binary Classification Report: \n", classification_report(y_holdout_hb,y_pred_hb))
    print("ROC AUC Score:", roc_auc_score(y_holdout_hb, y_prob_hb))

    cm_hb = confusion_matrix(y_holdout_hb, y_pred_hb)
    print("Confusion Matrix: \n", cm_hb)
    false_pos_false_neg(cm_hb)


    # Hashed + count validation
    y_pred_hc = loaded_model_hc.predict(x_holdout_hc)
    y_prob_hc = loaded_model_hc.predict_proba(x_holdout_hc)[:,1]

    print("\n Hashed + count Classification Report: \n", classification_report(y_holdout_hc,y_pred_hc))
    print("ROC AUC Score:", roc_auc_score(y_holdout_hc, y_prob_hc))

    cm_hc = confusion_matrix(y_holdout_hc, y_pred_hc)
    print("Confusion Matrix: \n", cm_hc)
    false_pos_false_neg(cm_hc)

    # Unhashed validation
    y_pred_uh = loaded_model_uh.predict(x_holdout_uh)
    y_prob_uh = loaded_model_uh.predict_proba(x_holdout_uh)[:,1]

    print("\n Unhashed Classification Report: \n", classification_report(y_holdout_uh,y_pred_uh))
    print("ROC AUC Score:", roc_auc_score(y_holdout_uh, y_prob_uh))

    cm_uh = confusion_matrix(y_holdout_uh, y_pred_uh)
    print("Confusion Matrix: \n", cm_uh)
    false_pos_false_neg(cm_uh)

if __name__ == "__main__":
    main()

