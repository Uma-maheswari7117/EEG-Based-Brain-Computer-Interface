
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ==================================
# LOAD DATA
# ==================================

data = pd.read_csv("../data/features.csv")

print("Dataset loaded!")
print("Total samples:", len(data))


# ==================================
# FEATURES AND LABEL
# ==================================

X = data[
    [
        "delta",
        "theta",
        "alpha",
        "beta",
        "mean",
        "std"
    ]
]

y = data["label"]


# ==================================
# TRAIN / TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================
# CREATE MODEL
# ==================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==================================
# TRAIN
# ==================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ==================================
# PREDICTION
# ==================================

y_pred = model.predict(X_test)


# ==================================
# ACCURACY
# ==================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nModel Accuracy:")

print(
    f"{accuracy * 100:.2f}%"
)


# ==================================
# CLASSIFICATION REPORT
# ==================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Relax",
            "Focus"
        ]
    )
)


# ==================================
# CONFUSION MATRIX
# ==================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==================================
# SAVE MODEL
# ==================================

model_path = "../model/eeg_model.pkl"

joblib.dump(
    model,
    model_path
)


print("\nModel saved successfully!")

print(
    "Location:",
    model_path
)