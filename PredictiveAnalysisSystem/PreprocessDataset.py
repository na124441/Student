
import pandas as pd
import numpy as np
import json

from pathlib import Path


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("student_academic_performance.csv")


# ==========================================
# 2. CONFIGURATION
# ==========================================

TARGET = "final_exam_score"

NUMERIC_COLS = [
    "age",
    "attendance_percentage",
    "study_hours_per_day",
    "previous_exam_score",
    "assignment_completion",
    "sleep_hours",
    "extracurricular_hours",
    "family_income"
]

BINARY_COLS = ["internet_access"]

CATEGORICAL_COLS = ["gender"]


# ==========================================
# 3. BINARY ENCODING
# ==========================================

def binary_encode(series):

    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
        "1.0": 1.0,
        "0.0": 0.0
    }

    encoded = series.astype(str).str.strip().str.lower().map(mapping)

    if encoded.isna().any():
        raise ValueError(f"Invalid binary values in {series.name}")

    return encoded.astype(float)


for col in BINARY_COLS:
    data[col] = binary_encode(data[col])


# ==========================================
# 4. NUMERIC CONVERSION
# ==========================================

for col in NUMERIC_COLS + [TARGET]:
    data[col] = pd.to_numeric(data[col], errors="raise")


# ==========================================
# 5. ONE-HOT ENCODING
# ==========================================

def one_hot_encode(df, columns):

    df = df.copy()
    categories = {}

    for col in columns:

        unique_values = sorted(df[col].dropna().unique())

        categories[col] = unique_values

        for value in unique_values:
            new_col = f"{col}_{value}"

            df[new_col] = (df[col] == value).astype(float)

        df.drop(columns=[col], inplace=True)

    return df, categories


# ==========================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

def dataset_split(df, train_ratio=0.70, val_ratio=0.15):

    if not (0 < train_ratio < 1):
        raise ValueError("Invalid train ratio")

    if not (0 < val_ratio < 1):
        raise ValueError("Invalid validation ratio")

    if train_ratio + val_ratio >= 1:
        raise ValueError("Train + validation must be less than 1")

    # Shuffle dataset
    shuffled = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    total = len(shuffled)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = shuffled.iloc[:train_end].copy()
    val = shuffled.iloc[train_end:val_end].copy()
    test = shuffled.iloc[val_end:].copy()

    return train, val, test


train, val, test = dataset_split(data)


# ==========================================
# 7. MIN-MAX NORMALIZATION
# ==========================================

def fit_minmax(train_df, columns):

    parameters = {}

    for col in columns:

        minimum = train_df[col].min()
        maximum = train_df[col].max()

        parameters[col] = {
            "min": float(minimum),
            "max": float(maximum)
        }

    return parameters


def transform_minmax(df, parameters):

    df = df.copy()

    for col, params in parameters.items():

        minimum = params["min"]
        maximum = params["max"]

        if maximum == minimum:
            df[col] = 0.0
        else:
            df[col] = (
                (df[col] - minimum) /
                (maximum - minimum)
            )

    return df


# ==========================================
# 8. SEPARATE FEATURES AND TARGET
# ==========================================

def separate_features_target(df):

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    return X, y


X_train, y_train = separate_features_target(train)
X_val, y_val = separate_features_target(val)
X_test, y_test = separate_features_target(test)


# ==========================================
# 9. FIT ENCODING ON TRAINING DATA
# ==========================================

X_train, category_parameters = one_hot_encode(
    X_train,
    CATEGORICAL_COLS
)

# Apply exactly the same categories to validation
# and test datasets.

def apply_one_hot(df, categories):

    df = df.copy()

    for col, unique_values in categories.items():

        for value in unique_values:

            new_col = f"{col}_{value}"

            df[new_col] = (
                df[col] == value
            ).astype(float)

        df.drop(columns=[col], inplace=True)

    return df


X_val = apply_one_hot(X_val, category_parameters)
X_test = apply_one_hot(X_test, category_parameters)


# ==========================================
# 10. FIT NORMALIZATION ON TRAIN DATA
# ==========================================

normalization_parameters = fit_minmax(
    X_train,
    NUMERIC_COLS
)

X_train = transform_minmax(
    X_train,
    normalization_parameters
)

X_val = transform_minmax(
    X_val,
    normalization_parameters
)

X_test = transform_minmax(
    X_test,
    normalization_parameters
)


# ==========================================
# 11. RECONSTRUCT DATASETS
# ==========================================

train_processed = X_train.assign(
    final_exam_score=y_train
)

val_processed = X_val.assign(
    final_exam_score=y_val
)

test_processed = X_test.assign(
    final_exam_score=y_test
)


# ==========================================
# 12. SAVE DATASETS
# ==========================================

output_dir = Path("dataset")
output_dir.mkdir(parents=True, exist_ok=True)

train_processed.to_csv(
    output_dir / "train.csv",
    index=False
)

val_processed.to_csv(
    output_dir / "val.csv",
    index=False
)

test_processed.to_csv(
    output_dir / "test.csv",
    index=False
)


# ==========================================
# 13. SAVE PREPROCESSING PARAMETERS
# ==========================================

metadata = {
    "normalization": normalization_parameters,
    "categories": category_parameters
}

with open(output_dir / "preprocessing_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)


# ==========================================
# 14. VERIFICATION
# ==========================================

print("\nPreprocessing completed successfully!")

print("\nDataset sizes:")
print("Train:", train_processed.shape)
print("Validation:", val_processed.shape)
print("Test:", test_processed.shape)

print("\nMissing values:")
print("Train:", train_processed.isnull().sum().sum())
print("Validation:", val_processed.isnull().sum().sum())
print("Test:", test_processed.isnull().sum().sum())

print("\nSaved files:")
print("dataset/train.csv")
print("dataset/val.csv")
print("dataset/test.csv")
print("dataset/preprocessing_metadata.json")
