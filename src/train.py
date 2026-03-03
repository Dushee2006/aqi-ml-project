import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocessing import load_and_preprocess


def train_model():
    # Load preprocessed data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # Create Logistic Regression model
    model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler


if __name__ == "__main__":
    model, scaler = train_model()

    # Save model and scaler
    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModel and scaler saved successfully.")