import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
from preprocessing import load_and_preprocess


def generate_reports():
    os.makedirs("reports", exist_ok=True)

    (
        X_train_unscaled,
        X_test_unscaled,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    ) = load_and_preprocess()

    # Load models
    log_model = joblib.load("models/logistic_model.pkl")
    rf_model = joblib.load("models/random_forest_model.pkl")

    # ===============================
    # Predictions
    # ===============================
    y_pred_log = log_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_unscaled)

    # ===============================
    # Accuracy + F1
    # ===============================
    report_log = classification_report(y_test, y_pred_log, output_dict=True)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    accuracy_log = accuracy_score(y_test, y_pred_log)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    macro_f1_log = report_log["macro avg"]["f1-score"]
    macro_f1_rf = report_rf["macro avg"]["f1-score"]

    # Save performance table
    performance_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [accuracy_log, accuracy_rf],
        "Macro F1 Score": [macro_f1_log, macro_f1_rf]
    })

    performance_df.to_csv("reports/model_performance.csv", index=False)

    # ===============================
    # Accuracy Comparison Plot
    # ===============================
    plt.figure()
    plt.bar(["Logistic", "Random Forest"], [accuracy_log, accuracy_rf])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.savefig("reports/accuracy_comparison.png")
    plt.close()

    # ===============================
    # Macro F1 Comparison Plot
    # ===============================
    plt.figure()
    plt.bar(["Logistic", "Random Forest"], [macro_f1_log, macro_f1_rf])
    plt.title("Model Macro F1 Comparison")
    plt.ylabel("Macro F1 Score")
    plt.savefig("reports/f1_comparison.png")
    plt.close()

    # ===============================
    # Confusion Matrices
    # ===============================
    cm_log = confusion_matrix(y_test, y_pred_log)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    plt.figure()
    sns.heatmap(cm_log, annot=True, fmt="d")
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig("reports/confusion_matrix_logistic.png")
    plt.close()

    plt.figure()
    sns.heatmap(cm_rf, annot=True, fmt="d")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("reports/confusion_matrix_rf.png")
    plt.close()

    # ===============================
    # Original Class Distribution
    # ===============================
    df_original = pd.read_csv("data/aqi_data.csv")

    plt.figure()
    df_original["aqi"].value_counts().sort_index().plot(kind="bar")
    plt.title("Original AQI Class Distribution")
    plt.savefig("reports/original_class_distribution.png")
    plt.close()

    # ===============================
    # Balanced Training Distribution
    # ===============================
    plt.figure()
    pd.Series(y_train).value_counts().sort_index().plot(kind="bar")
    plt.title("Balanced Training Distribution (After SMOTE)")
    plt.savefig("reports/balanced_training_distribution.png")
    plt.close()

    print("Advanced evaluation reports generated successfully!")


if __name__ == "__main__":
    generate_reports()