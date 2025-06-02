import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def pretty_confusion_matrix(estimator, x, y_true, labels, figsize=(10,10),cmap='Blues', cell_fontsize=30):
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator,
        x,
        y_true,
        display_labels=labels,
        ax=ax,
        colorbar=False,
        cmap=cmap
    )

    # Format tick labels and axis spines
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # Axis labels
    ax.set_xlabel("Predicted mutagenicity", fontsize=32, labelpad=30)
    ax.set_ylabel("True mutagenicity", fontsize=32, labelpad=20)
    ax.tick_params(axis='both', which='major', width=3, length=10)
    # Update font size of text inside matrix cells
    if hasattr(disp, 'text_'):
        for row in disp.text_:
            for text_obj in row.flat:
                text_obj.set_fontsize(cell_fontsize)

    plt.tight_layout()
    plt.show()


def pretty_ROC_AUC(estimator, x_test, y_test, model_name="Model", color="#004B7D", baseline_color="#356B90"):
    y_probs = estimator.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(32, 24))
    plt.plot(fpr, tpr, color=color, lw=18, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color=baseline_color, lw=18, linestyle='--', label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=75, length=25, width=8)
    plt.tick_params(axis='both', which='minor', length=15, width=10)
    plt.xlabel("False Positive Rate", fontsize=75, labelpad=40)
    plt.ylabel("True Positive Rate", fontsize=75, labelpad=40)
    plt.title(f"ROC Curve: {model_name}", fontsize=50, pad=50)
    plt.tick_params(axis='both', which='major', labelsize=75)
    plt.legend(loc="lower right", fontsize=60)
    plt.grid(False)
    ax = plt.gca()  # Get current Axes
    for spine in ax.spines.values():
        spine.set_linewidth(8)

    plt.tight_layout()
    plt.show()

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
    # Loading files
    training_uh = pd.read_csv("training_ames_data_uh.csv")
    training_hb = pd.read_csv("training_ames_data_hb.csv")
    training_hc = pd.read_csv("training_ames_data_hc.csv")

    # Preparing x and y data
    x_uh = training_uh.drop(columns=['AMES'])
    y_uh = training_uh['AMES']

    x_hb = training_hb.drop(columns=['AMES'])
    y_hb = training_hb['AMES']

    x_hc = training_hc.drop(columns=['AMES'])
    y_hc = training_hc['AMES']


    # UNHASHED + count data random forest classifier
    x_train, x_test, y_train, y_test = train_test_split(
        x_uh, y_uh, test_size=0.2, random_state=42, stratify=y_uh)

    rfuh = RandomForestClassifier(n_estimators=100, random_state=42)
    rfuh.fit(x_train, y_train)

    y_pred = rfuh.predict(x_test)
    print('\n Unhashed results \n ----------------')
    print(classification_report(y_test, y_pred))
    print('\n AUC', roc_auc_score(y_test, rfuh.predict_proba(x_test)[:, 1]))

    training_uh_cm = confusion_matrix(y_test, y_pred, labels=rfuh.classes_)
    print("\nConfusion Matrix:\n",training_uh_cm)

    pretty_confusion_matrix(estimator=rfuh, x=x_test, y_true=y_test, labels=rfuh.classes_)
    pretty_ROC_AUC(rfuh, x_test, y_test, model_name="Unhashed random forest classifier")
    false_pos_false_neg(cm=training_uh_cm)

    # HASHED + binary data random forest classifier
    x_train, x_test, y_train, y_test = train_test_split(
         x_hb, y_hb, test_size=0.2, random_state=42, stratify=y_hb)

    rfhb = RandomForestClassifier(n_estimators=100, random_state=42)
    rfhb.fit(x_train, y_train)

    y_pred = rfhb.predict(x_test)
    print('\n Hashed binary results')
    print('----------------')
    print(classification_report(y_test, y_pred))
    print('AUC', roc_auc_score(y_test, rfhb.predict_proba(x_test)[:, 1]))

    training_hb_cm = confusion_matrix(y_test, y_pred, labels=rfhb.classes_)
    print("\nConfusion Matrix:\n", training_hb_cm)

    pretty_confusion_matrix(estimator=rfhb,x=x_test,y_true=y_test,labels=rfhb.classes_)
    pretty_ROC_AUC(rfhb, x_test, y_test, model_name="Hashed binary random forest classifier")
    false_pos_false_neg(cm=training_hb_cm)

    #HASHED + count DATA random forest CLASSIFIER
    x_train, x_test, y_train, y_test = train_test_split(
        x_hc, y_hc, test_size=0.2, random_state=42, stratify=y_hc)

    rfhc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfhc.fit(x_train, y_train)

    y_pred = rfhc.predict(x_test)
    print('\n Hashed count results')
    print('----------------')
    print(classification_report(y_test, y_pred))
    print('AUC', roc_auc_score(y_test, rfhc.predict_proba(x_test)[:, 1]))

    training_hc_cm = confusion_matrix(y_test, y_pred, labels=rfhc.classes_)
    print("\nConfusion Matrix:\n", training_hc_cm)

    pretty_confusion_matrix(estimator=rfhc, x=x_test,y_true=y_test,labels=rfhc.classes_)
    pretty_ROC_AUC(rfhc, x_test, y_test, model_name="Hashed count random forest classifier")
    false_pos_false_neg(cm=training_hc_cm)

if __name__ == "__main__":
    main()
