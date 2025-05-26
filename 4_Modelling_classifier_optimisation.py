

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import joblib
import json

def main():
    # Loading files
    new_ames_hcv = pd.read_csv("ames_data_h_training.csv")
    new_ames_uhcv = pd.read_csv("ames_data_uh_training.csv")

    # Optimising HASHED DATA
    x_ames_hcv = new_ames_hcv.drop(columns=['AMES'])
    y_ames_hcv = new_ames_hcv['AMES'].values

    # Split data with stratification to preserve class balance
    xtrain, xtest, ytrain, ytest = train_test_split(
        x_ames_hcv, y_ames_hcv, test_size=0.15, random_state=42, stratify=y_ames_hcv
    )

    assert np.all(np.isfinite(xtrain)), "xtrain contains non-finite values."
    assert np.all(np.isfinite(ytrain)), "ytrain contains non-finite values."

    # Define the model
    rfh_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None, 'sqrt', 'log2']
    }

    # GridSearchCV with roc_auc scoring
    grid_search = GridSearchCV(
        estimator=rfh_classifier,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search
    grid_search.fit(xtrain, ytrain)

    # Best model and parameters
    best_model_hashed = grid_search.best_estimator_
    best_params_hashed = grid_search.best_params_

    # Predict
    ypred_test = best_model_hashed.predict(xtest)
    ypred_proba_test = best_model_hashed.predict_proba(xtest)[:, 1]  # Probabilities for the positive class

    # Evaluation
    print('Classification optimisation for hashed data')
    print("Best Parameters:", best_params_hashed)
    print("Best model score:", best_model_hashed.score(xtest, ytest))
    print("\nClassification Report:\n", classification_report(ytest, ypred_test))
    print("\nConfusion Matrix:\n", confusion_matrix(ytest, ypred_test))
    print("\nTest ROC AUC:", roc_auc_score(ytest, ypred_proba_test))

    joblib.dump(best_model_hashed, 'best_model_hashed.joblib')
    with open('best_rf_params_hashed.json', 'w') as f:
        json.dump(best_params_hashed, f)

    # Plot ROC Curve
    plt.rcParams.update({
        'axes.linewidth': 8,  # Thicker axis lines
        'xtick.major.width': 8,  # Thicker ticks
        'ytick.major.width': 8,
        'legend.fontsize': 50,  # Bigger legend font
        'xtick.major.size': 30,
        'ytick.major.size': 30,
        'ytick.minor.size': 10,
        'xtick.minor.size': 10,
        'xtick.labelsize': 55,
        'ytick.labelsize': 55
    })

    fpr, tpr, _ = roc_curve(ytest, ypred_proba_test)
    plt.figure(figsize=(32, 24))
    plt.plot(fpr, tpr, color = '#433259', lw=15, label=f"ROC Curve (AUC = {roc_auc_score(ytest, ypred_proba_test):.2f})")
    plt.plot([0, 1], [0, 1], color='#644B86', lw=15, linestyle='--', label="Random Classifier")
    plt.xlabel("False Positive Rate",fontsize=55, labelpad=40)
    plt.ylabel("True Positive Rate",fontsize=55, labelpad=40)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Roc Curve for hashed fingerprints (optimised)", fontsize=60, pad=50)
    plt.legend(loc="lower right")
    plt.grid(False)

    plt.show()


    # Optimising UNHASHED DATA
    x_ames_uhcv = new_ames_uhcv.drop(columns=['AMES'])
    y_ames_uhcv = new_ames_uhcv['AMES'].values

    # Split data with stratification to preserve class balance
    xtrain, xtest, ytrain, ytest = train_test_split(
        x_ames_uhcv, y_ames_uhcv, test_size=0.15, random_state=42, stratify=y_ames_uhcv
    )

    assert np.all(np.isfinite(xtrain)), "xtrain contains non-finite values."
    assert np.all(np.isfinite(ytrain)), "ytrain contains non-finite values."

    # Define the model
    rfuh_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None, 'sqrt', 'log2']
    }

    # GridSearchCV with roc_auc scoring
    grid_search = GridSearchCV(
        estimator=rfuh_classifier,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search
    grid_search.fit(xtrain, ytrain)

    # Best model and parameters
    best_model_unhashed = grid_search.best_estimator_
    best_params_unhashed = grid_search.best_params_

    # Predict
    ypred_test = best_model_unhashed.predict(xtest)
    ypred_proba_test = best_model_unhashed.predict_proba(xtest)[:, 1]  # Probabilities for the positive class

    # Evaluation
    print('Classification optimisation for hashed data')
    print("Best Parameters:", best_params_unhashed)
    print("\nClassification Report:\n", classification_report(ytest, ypred_test))
    print("\nConfusion Matrix:\n", confusion_matrix(ytest, ypred_test))
    print("\nTest ROC AUC:", roc_auc_score(ytest, ypred_proba_test))

    joblib.dump(best_model_unhashed, 'best_model_unhashed.joblib')
    with open('best_rf_params_unhashed.json', 'w') as f:
        json.dump(best_params_unhashed, f)


    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(ytest, ypred_proba_test)
    plt.figure(figsize=(32, 24))
    plt.plot(fpr, tpr, color = '#433259', lw=15, label=f"ROC Curve (AUC = {roc_auc_score(ytest, ypred_proba_test):.2f})")
    plt.plot([0, 1], [0, 1], color='#644B86', lw=15, linestyle='--', label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=55, labelpad=40)
    plt.ylabel("True Positive Rate", fontsize=55, labelpad=40)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("ROC Curve for unhashed fingerprints (Optimised)", fontsize=60, pad=50)
    plt.grid(False)
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()






