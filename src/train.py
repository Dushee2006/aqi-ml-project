import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import load_and_preprocess


def train_models():
    (
        X_train_unscaled,
        X_test_unscaled,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = load_and_preprocess()

    print("\n==============================")
    print("LOGISTIC REGRESSION")
    print("==============================")

    log_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print(classification_report(y_test, y_pred_log))

    print("\n==============================")
    print("RANDOM FOREST")
    print("==============================")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ) 

    rf_model.fit(X_train_unscaled, y_train)



    # ===============================
    # Normal Random Forest Evaluation
    # ===============================

    y_pred_rf = rf_model.predict(X_test_unscaled)

    print("\nRandom Forest Accuracy:",
          accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    return log_model, rf_model, scaler


if __name__ == "__main__":
    log_model, rf_model, scaler = train_models()

    joblib.dump(log_model, "models/logistic_model.pkl")
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModels and scaler saved successfully.")