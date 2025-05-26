
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import joblib
import json
import pandas as pd

def main():
    # importing the two models
    loaded_model_h = joblib.load("best_model_hashed.joblib")
    loaded_model_uh = joblib.load("best_model_unhashed.joblib")
    print("Models loaded successfully.")

    # Loading files
    holdout_uh = pd.read_csv("ames_data_uh_validation.csv")
    holdout_h = pd.read_csv("ames_data_h_validation.csv")

    # Specifying HASHED holdout data
    x_holdout_h = holdout_h.drop(columns=['AMES'])
    y_holdout_h = holdout_h['AMES']

    # Specifying UNHASHED holdout data
    x_holdout_uh = holdout_uh.drop(columns=['AMES'])
    y_holdout_uh = holdout_uh['AMES']

    # Use the loaded model to make predictions on the holdout test set
    y_pred_h = loaded_model_h.predict(x_holdout_h)
    y_prob_h = loaded_model_h.predict_proba(x_holdout_h)[:,1]

    print("Hashed Classification Report: \n", classification_report(y_holdout_h,y_pred_h))
    print("ROC AUC Score:", roc_auc_score(y_holdout_h, y_prob_h))

    cm_h = confusion_matrix(y_holdout_h, y_pred_h)
    print("Confusion Matrix: \n", cm_h)

    # Doing it again for unhashed
    # This time it doesn't work... I am not sure why
    y_pred_uh = loaded_model_uh.predict(x_holdout_uh)
    y_prob_uh = loaded_model_uh.predict_proba(x_holdout_uh)[:,1]

    print("Unhashed Classification Report: \n", classification_report(y_holdout_uh,y_pred_uh))
    print("ROC AUC Score:", roc_auc_score(y_holdout_uh, y_prob_uh))

    cm_uh = confusion_matrix(y_holdout_uh, y_pred_uh)
    print("Confusion Matrix: \n", cm_uh)

if __name__ == "__main__":
    main()

