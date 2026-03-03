import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess(filepath="data/aqi_data.csv"):
    # Load dataset
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Drop original timestamp
    df = df.drop(columns=["timestamp"])

    # Separate features and target
    X = df.drop(columns=["aqi"])
    y = df["aqi"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_resampled,   # Unscaled training (for Random Forest)
        X_test,              # Unscaled test
        X_train_scaled,      # Scaled training (for Logistic)
        X_test_scaled,       # Scaled test
        y_train_resampled,
        y_test,
        scaler
    )