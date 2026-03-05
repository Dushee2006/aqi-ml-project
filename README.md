# AQI Classification ML Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-success)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

A production-ready Machine Learning system that predicts Air Quality Index (AQI) levels using pollutant measurements and time-based features.

This project demonstrates a complete ML workflow including data ingestion, preprocessing, model training, evaluation, and deployment.

---

# Live Application

Streamlit App

https://aqi-ml-project-wiqqndruisijceapaz5vpf.streamlit.app/

Users can input pollutant values and receive real-time AQI predictions using the trained Random Forest model.

---

# Dataset

Source: OpenWeather Historical AQI API  
Location: Colombo, Sri Lanka  
Total Records: 2760  

Target variable: AQI class (1–4)

| AQI | Air Quality |
|----|-------------|
| 1 | Good |
| 2 | Fair |
| 3 | Moderate |
| 4 | Poor |

### Features Used

- pm2_5
- pm10
- co
- no2
- o3
- so2
- hour
- day
- month
- day_of_week

---

# Data Preprocessing Pipeline

The preprocessing pipeline was carefully designed to avoid data leakage and improve model performance.

Steps included:

1. Timestamp conversion to datetime
2. Time-based feature extraction (hour, day, month, weekday)
3. Train-test split (80/20 stratified)
4. Missing value handling
5. Outlier treatment using IQR clipping
6. Handling class imbalance using SMOTE
7. Feature scaling using StandardScaler (for Logistic Regression)

---

# Handling Class Imbalance

The original dataset contained class imbalance.

SMOTE was applied only to the training data to avoid data leakage.

Original Dataset Distribution

![Original Distribution](reports/original_class_distribution.png)

Balanced Training Data

![Balanced Distribution](reports/balanced_training_distribution.png)

---

# Models Trained

Two models were implemented and compared.

Logistic Regression  
A baseline linear classification model.

Random Forest Classifier  
An ensemble tree-based model capable of learning complex non-linear relationships.

---

# Model Evaluation

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1 Score
- Confusion Matrix

---

# Model Performance Comparison

| Model | Accuracy | Macro F1 Score |
|------|---------|----------------|
| Logistic Regression | 0.906 | 0.916 |
| Random Forest | 0.998 | 0.999 |

Random Forest clearly outperformed Logistic Regression across all evaluation metrics.

---

# Accuracy Comparison

![Accuracy Comparison](reports/accuracy_comparison.png)

---

# Macro F1 Score Comparison

![F1 Comparison](reports/f1_comparison.png)

---

# Confusion Matrices

Logistic Regression

![Confusion Matrix Logistic](reports/confusion_matrix_logistic.png)

Random Forest

![Confusion Matrix Random Forest](reports/confusion_matrix_rf.png)

---

# Random Forest Feature Importance

![Feature Importance](reports/feature_importance_rf.png)

The most influential pollutants in AQI prediction were:

- PM2.5
- PM10
- O3

---

# Data Leakage Prevention

Several steps were taken to ensure a reliable evaluation:

- Train-test split performed before SMOTE
- SMOTE applied only to training data
- Test data kept untouched
- Shuffle test performed to confirm model learning

Shuffle test accuracy dropped to ~0.27, confirming the absence of data leakage.

---

# Final Model Selection

Random Forest was selected as the final production model because it:

- Achieved higher accuracy
- Achieved higher Macro F1 score
- Performed better on minority classes
- Captured non-linear pollutant interactions effectively

---

# Deployment

The trained model was deployed using Streamlit Cloud.

Deployment workflow:

1. Model saved using joblib
2. Streamlit UI created in app.py
3. GitHub repository connected
4. Streamlit Cloud automatically builds and deploys the application

---

# Project Structure

```
aqi_ml_project
│
├── data
│   └── aqi_data.csv
│
├── models
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── reports
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_rf.png
│   ├── feature_importance_rf.png
│   ├── original_class_distribution.png
│   ├── balanced_training_distribution.png
│   └── model_performance.csv
│
├── src
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Tech Stack

- Python
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Git
- GitHub

---

# Key Learnings

This project strengthened knowledge in:

- Handling imbalanced classification datasets
- Preventing data leakage in ML pipelines
- Model comparison using macro evaluation metrics
- Feature importance analysis
- Building production-ready ML systems
- Deploying ML applications using Streamlit Cloud

---

If you found this project interesting, consider starring the repository.