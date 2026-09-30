# ============================================================
# DEEP LEARNING PROJECT
# DIABETIC RETINOPATHY PREDICTION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib


print("Libraries imported successfully!")

# Show TensorFlow version
print("TensorFlow version:", tf.__version__)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "P600_pronostico_dataset.xls"

df = pd.read_csv(
    file_path,
    sep=";"
)

print("\n========================================")
print("DATASET LOADED")
print("========================================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. UNDERSTAND DATASET
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\n========================================")
print("DATA CLEANING")
print("========================================")

# Check missing values

print("\nMissing values:")
print(df.isnull().sum())


# Check duplicate rows

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Remove duplicates

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# Check infinite values

numeric_columns = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol"
]

print("\nInfinite values:")
print(
    np.isinf(
        df[numeric_columns]
    ).sum()
)


# ============================================================
# 5. TARGET ANALYSIS
# ============================================================

print("\n========================================")
print("TARGET ANALYSIS")
print("========================================")

print("\nTarget values:")
print(df["prognosis"].unique())

print("\nTarget distribution:")
print(df["prognosis"].value_counts())


# ============================================================
# 6. EDA - TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="prognosis"
)

plt.title("Prognosis Distribution")
plt.xlabel("Prognosis")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ============================================================
# 7. EDA - AGE
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# ============================================================
# 8. EDA - SYSTOLIC BP
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="systolic_bp",
    bins=30,
    kde=True
)

plt.title("Systolic Blood Pressure Distribution")
plt.xlabel("Systolic BP")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# ============================================================
# 9. EDA - DIASTOLIC BP
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="diastolic_bp",
    bins=30,
    kde=True
)

plt.title("Diastolic Blood Pressure Distribution")
plt.xlabel("Diastolic BP")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# ============================================================
# 10. EDA - CHOLESTEROL
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="cholesterol",
    bins=30,
    kde=True
)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# ============================================================
# 11. OUTLIER ANALYSIS
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df[numeric_columns]
)

plt.title("Numerical Features - Boxplot")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


# ============================================================
# 12. CORRELATION HEATMAP
# ============================================================

correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 13. FEATURES AND TARGET
# ============================================================

print("\n========================================")
print("FEATURE SELECTION")
print("========================================")

# ID is NOT used because it is only an identifier.

X = df[
    [
        "age",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol"
    ]
]

y = df["prognosis"]


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 14. ENCODE TARGET
# ============================================================

print("\n========================================")
print("TARGET ENCODING")
print("========================================")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nTarget mapping:")

mapping = dict(
    zip(
        label_encoder.classes_,
        label_encoder.transform(
            label_encoder.classes_
        )
    )
)

print(mapping)


# ============================================================
# 15. TRAIN TEST SPLIT
# ============================================================

print("\n========================================")
print("TRAIN TEST SPLIT")
print("========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 16. FEATURE SCALING
# ============================================================

print("\n========================================")
print("FEATURE SCALING")
print("========================================")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nScaling completed!")

print("\nFirst 5 scaled training records:")
print(X_train[:5])


# ============================================================
# 17. BUILD ANN MODEL
# ============================================================

print("\n========================================")
print("BUILDING ANN MODEL")
print("========================================")

model = Sequential()


# Input + First Hidden Layer
model.add(
    Dense(
        16,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)


# Dropout
model.add(
    Dropout(0.2)
)


# Second Hidden Layer
model.add(
    Dense(
        8,
        activation="relu"
    )
)


# Output Layer
model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ============================================================
# 18. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 19. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 20. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 21. TRAIN MODEL
# ============================================================

print("\n========================================")
print("TRAINING ANN")
print("========================================")

history = model.fit(
    X_train,
    y_train,

    epochs=100,

    batch_size=32,

    validation_split=0.20,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# 22. TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 23. TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 24. MODEL PREDICTION
# ============================================================

print("\n========================================")
print("MODEL PREDICTION")
print("========================================")

y_probability = model.predict(
    X_test
)

# Convert probability to 0 or 1

y_pred = (
    y_probability >= 0.5
).astype(int).flatten()


print("\nFirst 10 predicted values:")
print(y_pred[:10])

print("\nFirst 10 actual values:")
print(y_test[:10])


# ============================================================
# 25. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("MODEL ACCURACY")
print("========================================")

print(
    "Test Accuracy:",
    accuracy
)

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 28. SAVE MODEL
# ============================================================

model.save(
    "model.keras"
)

print("\nDeep learning model saved as model.keras")


# ============================================================
# 29. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler saved as scaler.pkl")


# ============================================================
# 30. SAVE LABEL ENCODER
# ============================================================

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("Label encoder saved as label_encoder.pkl")


# ============================================================
# 31. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("PROJECT COMPLETED")
print("========================================")

print("Data cleaning       : Completed")
print("EDA                 : Completed")
print("Preprocessing       : Completed")
print("ANN model           : Completed")
print("Model training      : Completed")
print("Model evaluation    : Completed")
print("Model saving        : Completed")

print("\nReady for Streamlit deployment!")
