"""
Train and save the same pipeline as the notebook:
StandardScaler -> SVC(kernel='linear'), train_test_split(..., random_state=2).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "pima_raw.csv"
OUT_PATH = BASE / "model" / "artifacts.joblib"

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def main():
    diabetes_dataset = pd.read_csv(DATA_PATH, names=COLUMNS)
    X = diabetes_dataset.drop(columns="Outcome", axis=1)
    Y = diabetes_dataset["Outcome"]

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = diabetes_dataset["Outcome"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )

    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, Y_train)

    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    print("Accuracy score of the training data:", training_data_accuracy)
    print("Accuracy score of the test data:", test_data_accuracy)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "classifier": classifier,
            "feature_names": COLUMNS[:-1],
            "metrics": {
                "train_accuracy": float(training_data_accuracy),
                "test_accuracy": float(test_data_accuracy),
            },
        },
        OUT_PATH,
    )
    print(f"Saved model to {OUT_PATH}")


if __name__ == "__main__":
    main()
