"""
RL Based DDoS Threat Detection in Cloud Environment
---------------------------------------------------
train.py
This script trains multiple ML/DL classifiers for DDoS detection using either
UNSW-NB15 or CICDDoS2019 dataset.

Usage:
    python train.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------
def load_dataset(path, label_col="Label"):
    """Load dataset CSV and split features/labels."""
    df = pd.read_csv(path)
    X = df.drop(columns=[label_col])
    y = df[label_col].map(lambda v: 1 if v in ["Attack", 1] else 0)  # binary: Attack=1, Benign=0
    return X, y

# ------------------------------------------------------------------
# Prepare Data
# ------------------------------------------------------------------
def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------
# Evaluation Helper
# ------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:  # SVM fallback
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results.append([name, acc, prec, rec, f1, auc])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return fpr, tpr

# ------------------------------------------------------------------
# ML Models
# ------------------------------------------------------------------
def train_ml_models(X_train, X_test, y_train, y_test):
    results = []
    roc_curves = {}

    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Ensemble": VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
                ("nb", GaussianNB())
            ],
            voting="soft"
        ),
        "DNN": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=50, random_state=RANDOM_STATE)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fpr, tpr = evaluate_model(name, model, X_test, y_test, results)
        roc_curves[name] = (fpr, tpr)

    return results, roc_curves

# ------------------------------------------------------------------
# CNN & LSTM Models
# ------------------------------------------------------------------
def train_cnn_lstm(X_train, X_test, y_train, y_test):
    results = {}
    roc_curves = {}

    # Reshape
    X_train_r = np.expand_dims(X_train, axis=2)
    X_test_r = np.expand_dims(X_test, axis=2)

    # CNN
    cnn = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_train_r.shape[1], 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train_r, y_train, epochs=3, batch_size=128, verbose=1)
    y_prob = cnn.predict(X_test_r).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    results["CNN"] = [accuracy_score(y_test, y_pred),
                      precision_score(y_test, y_pred),
                      recall_score(y_test, y_pred),
                      f1_score(y_test, y_pred),
                      roc_auc_score(y_test, y_prob)]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves["CNN"] = (fpr, tpr)

    # LSTM
    lstm = Sequential([
        LSTM(64, input_shape=(X_train_r.shape[1], 1)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_r, y_train, epochs=3, batch_size=128, verbose=1)
    y_prob = lstm.predict(X_test_r).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    results["LSTM"] = [accuracy_score(y_test, y_pred),
                       precision_score(y_test, y_pred),
                       recall_score(y_test, y_pred),
                       f1_score(y_test, y_pred),
                       roc_auc_score(y_test, y_prob)]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves["LSTM"] = (fpr, tpr)

    return results, roc_curves

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Choose dataset: "UNSW" or "CIC"
    DATASET = "UNSW"

    if DATASET == "UNSW":
        DATA_PATH = "UNSW-NB15_features.csv"
        LABEL_COL = "Label"
    elif DATASET == "CIC":
        DATA_PATH = "cicddos2019_dataset.csv"
        LABEL_COL = "Label"

    print(f"\n=== Running on {DATASET} Dataset ===")
    X, y = load_dataset(DATA_PATH, label_col=LABEL_COL)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # ML models
    ml_results, ml_rocs = train_ml_models(X_train, X_test, y_train, y_test)

    # CNN & LSTM
    dl_results, dl_rocs = train_cnn_lstm(X_train, X_test, y_train, y_test)

    # Combine results
    results = pd.DataFrame(
        ml_results + [["CNN"] + dl_results["CNN"], ["LSTM"] + dl_results["LSTM"]],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    )
    print("\nFinal Results:\n", results)
    results.to_csv(f"results_summary_{DATASET}.csv", index=False)

    # ROC curves
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr) in {**ml_rocs, **dl_rocs}.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title(f"ROC Curve Comparison ({DATASET})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_comparison_{DATASET}.png", dpi=180)
    plt.show()
