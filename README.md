# churn-prediction
📌 Overview

The Customer Churn Prediction System is a desktop-based machine learning application that predicts whether a customer is likely to leave a service (churn) based on historical customer data. The application provides an offline, GUI-based solution where users can upload a CSV dataset and instantly receive predictions, evaluation metrics, and visual insights.

This project demonstrates a complete end-to-end data science pipeline, from data preprocessing and model training to deployment in a standalone desktop application.

🎯 Objectives

Predict customer churn using machine learning

Build a reusable and deployable ML model

Provide a non-technical, user-friendly GUI

Display evaluation metrics and visual explanations

Enable offline usage without any web framework

🧠 Problem Statement

Customer churn is a major concern for businesses, as acquiring new customers is often more expensive than retaining existing ones. By predicting churn in advance, companies can take proactive steps such as targeted offers or engagement strategies to retain customers.

This project aims to build a predictive system that identifies customers at risk of churn using historical data and presents the results in a clear and understandable format.

⚙️ System Workflow

User launches the desktop application

User uploads a CSV dataset

Application performs:

Data cleaning

Feature encoding and scaling

Feature alignment with training data

Pre-trained machine learning model is loaded

Predictions and probabilities are generated

Evaluation metrics and visualizations are displayed in a single dashboard

🛠️ Technologies Used
Programming Language

Python

Libraries

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib

Seaborn

Joblib

GUI Framework

Tkinter

Machine Learning Model

Random Forest Classifier

📁 Project Structure
churn/
│
├── data/
│   └── churn.csv
│
├── model/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── 1.py          # Model training & saving
├── gui_app.py    # Desktop GUI application
├── README.md

📊 Features

Upload CSV file directly through GUI

Automatic preprocessing and prediction

Handles class imbalance using SMOTE

Displays:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

ROC Curve

Feature Importance

All graphs and metrics shown simultaneously

Fully offline execution

▶️ How to Run the Project
Step 1: Install Required Libraries
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

Step 2: Train and Save the Model
python 1.py


This will generate:

model/random_forest.pkl
model/scaler.pkl
model/feature_columns.pkl

Step 3: Run the Desktop Application
python gui_app.py

📥 Input Requirements

CSV file with the same structure as the training dataset

Must include the target column Exited for evaluation

📈 Output

Churn prediction results

Classification metrics

Visual explanations for model decisions

Business-friendly insights

🧪 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC–AUC

🚀 Future Enhancements

Prediction-only mode (without target column)

Convert application to .exe

Save results as PDF or CSV

Add error handling for invalid datasets

Improve GUI layout with tabs or panels

💡 Learning Outcomes

Built a complete ML pipeline

Learned model persistence and reuse

Implemented ML deployment without web frameworks

Designed a user-friendly desktop application

Gained experience in real-world ML engineering challenges
