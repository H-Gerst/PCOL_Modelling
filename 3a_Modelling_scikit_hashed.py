import openpyxl
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import sklearn

from rdkit import Chem, DataStructs
from rdkit.Chem import Mol, PandasTools, AllChem, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

def main():
    # Loading files
    new_ames_h = pd.read_csv("ames_data_h_training.csv")

    # HASHED DATA random forest CLASSIFIER
    x_ames_hc = new_ames_h.drop(columns=['AMES'])
    y_ames_hc = new_ames_h['AMES']
    x_train, x_test, y_train, y_test = train_test_split(
         x_ames_hc, y_ames_hc, test_size=0.2, random_state=42, stratify=y_ames_hc)

    rfh = RandomForestClassifier(n_estimators=100, random_state=42)
    rfh.fit(x_train, y_train)

    y_pred = rfh.predict(x_test)
    print('\n Hashed results')
    print('----------------')
    print(classification_report(y_test, y_pred))
    print(roc_auc_score(y_test, rfh.predict_proba(x_test)[:, 1]))
    print(rfh.feature_importances_)

    # Plot ROC Curve for Hashed data
    y_probs_h = rfh.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs_h)
    roc_auc_h = auc(fpr, tpr)

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
    plt.figure(facecolor='#F5FBFF')
    plt.figure(figsize=(32, 24))
    plt.plot(fpr, tpr, color='#004B7D', lw=15, label=f"ROC curve (AUC = {roc_auc_h:.2f})")
    plt.plot([0, 1], [0, 1], color='#356B90', lw=15, linestyle='--', label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=55,labelpad=40)
    plt.ylabel("True Positive Rate", fontsize=55, labelpad=40)
    plt.title("ROC Curve: Random Forest Classifier for Hashed Data", fontsize=60, pad=50)
    plt.legend(loc="lower right")
    plt.grid(False)
    plt.show()



if __name__ == "__main__":
    main()

