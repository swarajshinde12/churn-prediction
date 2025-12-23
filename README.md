# Overview

The Customer Churn Prediction System is a  that predicts whether a customer is likely to leave a service using historical customer data. The system runs completely offline and provides results through a graphical user interface (GUI).

Users can upload a CSV dataset, after which the application automatically performs preprocessing, prediction, evaluation, and visualization using a pre-trained machine learning model.

# Objectives

Build an end-to-end machine learning solution

Predict customer churn using historical data

Provide an offline desktop-based interface

Display evaluation metrics and visual insights

Enable easy execution without web frameworks

# Problem Statement

Customer churn directly impacts business revenue and growth. Retaining existing customers is significantly more cost-effective than acquiring new ones. By identifying customers who are likely to churn, businesses can take proactive actions to reduce customer loss.

This project aims to develop a predictive system that identifies churn-prone customers and presents the results in a clear, visual, and understandable manner.

# System Workflow
Step 1

User launches the desktop application

Step 2

User uploads a CSV dataset

Step 3

Data preprocessing and feature engineering are applied

Step 4

Pre-trained machine learning model is loaded

Step 5

Predictions and probabilities are generated

Step 6

Evaluation metrics and visualizations are displayed on a single dashboard

# Technologies Used
## Programming Language

Python

## Libraries

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib

Seaborn

Joblib

## GUI Framework

Tkinter

## Machine Learning Model

Random Forest Classifier

# Project Structure
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
├── 1.py          # Model training and saving
├── gui_app.py    # Desktop GUI application
├── README.md

# Features

Desktop-based GUI for dataset upload

Automatic preprocessing and prediction

Handles class imbalance using SMOTE

Displays accuracy, precision, recall, and F1-score

Visualizations include confusion matrix, ROC curve, and feature importance

All metrics and graphs shown simultaneously

Fully offline and standalone execution

# How to Run the Project
## Step 1: Install Dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

## Step 2: Train and Save the Model
python 1.py


This step generates the trained model and preprocessing files inside the model directory.

## Step 3: Run the Desktop Application
python gui_app.py

# Input Requirements

CSV file with the same structure as the training dataset

Dataset must contain the target column Exited

# Output

Customer churn predictions

Accuracy, precision, recall, and F1-score

Confusion matrix visualization

ROC curve

Feature importance plot

# Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC–AUC

# Future Enhancements

Prediction-only mode without target column

Convert application to executable (.exe)

Export results to PDF or CSV

Improve GUI layout with tabs or panels

Add dataset validation and error handling

# Learning Outcomes

Built a complete end-to-end ML pipeline

Implemented model persistence and reuse

Developed an offline ML deployment solution

Gained experience with real-world ML engineering

Improved ML result interpretation using visualizations
