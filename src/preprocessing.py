import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess(filepath="data/aqi_data.csv"):
    # ======================================================
    # 1️⃣ LOAD DATA
    # ======================================================
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Basic validation
    if df.empty:
        raise ValueError("Dataset is empty!")

    print("Initial shape:", df.shape)

    # ======================================================
    # 2️⃣ FEATURE ENGINEERING
    # ======================================================
    print("Performing feature engineering...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df.drop(columns=["timestamp"], inplace=True)

    # ======================================================
    # 3️⃣ SEPARATE FEATURES & TARGET
    # ======================================================
    X = df.drop(columns=["aqi"])
    y = df["aqi"]

    # ======================================================
    # 4️⃣ TRAIN-TEST SPLIT (STRATIFIED)
    # ======================================================
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ======================================================
    # 5️⃣ MISSING VALUE HANDLING (TRAIN-BASED)
    # ======================================================
    print("Checking missing values...")

    missing_train = X_train.isnull().sum()
    missing_test = X_test.isnull().sum()

    if missing_train.sum() > 0:
        print("Missing values in training set:")
        print(missing_train[missing_train > 0])

    if missing_test.sum() > 0:
        print("Missing values in test set:")
        print(missing_test[missing_test > 0])

    # Compute medians ONLY from training set
    train_medians = X_train.median()

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # ======================================================
    # 6️⃣ OUTLIER HANDLING (IQR CLIPPING - TRAIN BASED)
    # ======================================================
    print("Handling outliers using IQR clipping...")

    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_train = X_train.clip(lower=lower_bound, upper=upper_bound, axis=1)
    X_test = X_test.clip(lower=lower_bound, upper=upper_bound, axis=1)

    # ======================================================
    # 7️⃣ HANDLE CLASS IMBALANCE (SMOTE - TRAIN ONLY)
    # ======================================================
    print("Applying SMOTE to training data...")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Balanced training shape:", X_train_resampled.shape)

    # ======================================================
    # 8️⃣ FEATURE SCALING (FOR LINEAR MODELS)
    # ======================================================
    print("Scaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing completed successfully.")

    # ======================================================
    # RETURN ALL NECESSARY OBJECTS
    # ======================================================
    return (
        X_train_resampled,   # For tree-based models
        X_test,
        X_train_scaled,      # For linear models
        X_test_scaled,
        y_train_resampled,
        y_test,
        scaler
    )