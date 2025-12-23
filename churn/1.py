import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import joblib

print("VS Code environment is ready")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/churn.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df['Exited'].value_counts())
print(df['Exited'].value_counts(normalize=True))

# ===============================
# CLEAN DATA
# ===============================
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
print(df.shape)

X = df.drop('Exited', axis=1)
y = df['Exited']

# ===============================
# FEATURE ENGINEERING
# ===============================
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# 🔴 IMPORTANT: SAVE FEATURE COLUMNS
feature_columns = X.columns

# ===============================
# SCALING
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Before SMOTE:")
print(y_train.value_counts())

# ===============================
# SMOTE
# ===============================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_smote.value_counts())

# ===============================
# MODELS
# ===============================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_smote, y_train_smote)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_smote, y_train_smote)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_smote, y_train_smote)

svm = SVC(probability=True)
svm.fit(X_train_smote, y_train_smote)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_smote, y_train_smote)

# ===============================
# EVALUATION
# ===============================
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate_model(rf, X_test, y_test, "Random Forest")

# ===============================
# SAVE MODEL FILES
# ===============================
os.makedirs("model", exist_ok=True)

joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_columns, "model/feature_columns.pkl")

print("Model, scaler, and feature columns saved successfully")
