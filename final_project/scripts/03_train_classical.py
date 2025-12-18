"""
Train and evaluate classical ML models using MediaPipe landmarks.
"""

import os
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def main():
    with open("models/classical/landmarks_train_unseen.pkl", "rb") as f:
        data = pickle.load(f)

    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"])
    X_test = np.array(data["X_unseen"], dtype=np.float32)
    y_test = np.array(data["y_unseen"])

    models = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=6))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ]),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=8, random_state=42
        )
    }

    os.makedirs("outputs/classical", exist_ok=True)
    summary = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        summary.append((name, acc))

        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

    with open("outputs/classical/accuracies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy"])
        for row in summary:
            writer.writerow(row)

if __name__ == "__main__":
    main()
