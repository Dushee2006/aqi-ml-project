# AQI Prediction Web App (Machine Learning)

## Overview
This project predicts Air Quality Index (AQI) levels using machine learning models trained on historical air pollution data.

The application is deployed using Streamlit Cloud and allows users to input pollutant values to predict AQI category.

## Live Demo
🔗 https://aqi-ml-project-wiqqndruisijceapaz5vpf.streamlit.app/

## Features
- Historical data ingestion from OpenWeather API
- Feature engineering (time-based features)
- Dataset balancing using SMOTE
- Logistic Regression and Random Forest models
- Model evaluation with classification report and confusion matrix
- Data leakage validation (shuffle test)
- Model persistence using joblib
- Deployed interactive web app using Streamlit

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Git & GitHub
- Streamlit Cloud Deployment

## Project Structure
```
aqi-ml-project/
│
├── app.py
├── requirements.txt
├── models/
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   └── train.py
```

## Model Performance
Random Forest achieved ~99.8% accuracy on test data.

## Author
Dulasha Bhanuki Wickramarathna Siriwardhana