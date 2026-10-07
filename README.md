# 🏠 House Price Prediction using Machine Learning

A Machine Learning project that predicts house prices based on property features.

## 📌 Project Overview

This project uses Machine Learning techniques to predict house prices based on different property-related features.

The project also includes a Flask web application where users can interact with the trained model.

## 🚀 Features

- 🏠 House price prediction
- 🤖 Machine Learning model
- 🌐 Flask web application
- 📊 Dataset-based prediction
- 💾 Trained model saved using Joblib

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML/CSS
- Jupyter Notebook

## 🤖 Machine Learning Models

This project evaluates multiple Machine Learning regression algorithms:

- 📈 Linear Regression
- 🌳 Decision Tree Regressor
- 🌲 Random Forest Regressor
- 🚀 Gradient Boosting Regressor

The models are trained and evaluated, and the best-performing model is selected for house price prediction.

## 🔄 Machine Learning Workflow

House Price Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Train Multiple Models
        ↓
Model Evaluation
        ↓
Select Best Model
        ↓
House Price Prediction
        ↓
Flask Web Application

## 📊 Model Performance

The models were evaluated using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

| Model | R² Score | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.7215 | $124,105.15 | $212,029.11 |
| Decision Tree | 0.5789 | $144,647.37 | $260,710.90 |
| Random Forest | 0.6995 | $113,697.28 | $220,232.30 |
| Gradient Boosting | 0.7165 | $115,509.80 | $213,912.58 |

### 🏆 Best Model

**Linear Regression**

- R² Score: **0.7215**
- MAE: **$124,105.15**
- RMSE: **$212,029.11**

The Linear Regression model achieved the highest R² score and was selected as the best-performing model for the final house price prediction.

## 📸 Project Screenshot

![House Price Prediction App](home.jpg)


## ▶️ How to Run

### 1. Clone the repository

git clone https://github.com/enoshsteve/House-Price-Prediction-ML.git

### 2. Open the project

cd House-Price-Prediction-ML

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the Flask application

python app\app.py

### 5. Open in browser

http://127.0.0.1:5000
