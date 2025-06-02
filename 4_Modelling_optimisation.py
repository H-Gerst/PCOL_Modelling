import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, ConfusionMatrixDisplay, \
    auc
import matplotlib.pyplot as plt
import joblib
import json


def optimise_and_evaluate_model(
    x, y,
    param_grid,
    model_name,
    model_filename,
    param_filename,
    color="#2f0c5c",
    baseline_color="#6f43a8"
):
    # Train/test/split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)


    assert np.all(np.isfinite(x_train)), "x_train contains non-finite values."
    assert np.all(np.isfinite(y_train)), "y_train contains non-finite values."

    # Model and grid search
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predictions
    y_pred = best_model.predict(x_test)
    y_proba = best_model.predict_proba(x_test)[:, 1]

    # Evaluation
    print(f'\nClassification optimisation for {model_name}')
    print("Best Parameters:", best_params)
    print("Best model score:", best_model.score(x_test, y_test))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n",cm)
    print("\nTest ROC AUC:", roc_auc_score(y_test, y_proba))

    false_pos_false_neg(cm)

    # Save model and parameters
    joblib.dump(best_model, model_filename)
    with open(param_filename, 'w') as f:
        json.dump(best_params, f)

    # Plots
    pretty_ROC_AUC(
        best_model,
        x_test,
        y_test,
        model_name=model_name,
        color=color,
        baseline_color=baseline_color
    )

    pretty_confusion_matrix(
        estimator=best_model,
        x=x_test,
        y_true=y_test,
        labels=best_model.classes_
    )


def pretty_confusion_matrix(estimator, x, y_true, labels, figsize=(10,10),cmap='Purples', cell_fontsize=30):
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


def pretty_ROC_AUC(refined_model, x_test, y_test, model_name="Model", color="#2f0c5c", baseline_color="#6f43a8"):
    y_probs = refined_model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(32, 24))
    plt.plot(fpr, tpr, color=color, lw=18, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color=baseline_color, lw=18, linestyle='--', label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=75, length=25, width=5)
    plt.tick_params(axis='both', which='minor', length=15, width=10)
    plt.xlabel("False Positive Rate", fontsize=75, labelpad=40)
    plt.ylabel("True Positive Rate", fontsize=75, labelpad=40)
    plt.title(f"ROC Curve: {model_name}", fontsize=60, pad=50)
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

    # Setting the paramaters to be searched
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None, 'sqrt', 'log2']
    }

    # UNHASHED
    optimise_and_evaluate_model(
        x=x_uh,
        y=y_uh,
        param_grid=param_grid,
        model_name= "Unhashed random forest classifier",
        model_filename= "best_model_unhashed.joblib",
        param_filename= "best_unhashed_params.joblib",
    )


    # HASHED, COUNT
    optimise_and_evaluate_model(
        x=x_hc,
        y=y_hc,
        param_grid=param_grid,
        model_name="Hashed count random forest classifier",
        model_filename="best_model_hashed_count.joblib",
        param_filename="best_hashed_count_params.joblib",
    )

    optimise_and_evaluate_model(
        x=x_hb,
        y=y_hb,
        param_grid=param_grid,
        model_name="Hashed binary random forest classifier",
        model_filename="best_model_hashed_binary.joblib",
        param_filename="best_hashed_binary_params.joblib",
    )


if __name__ == '__main__':
    main()





