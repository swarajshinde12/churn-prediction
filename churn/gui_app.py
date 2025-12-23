import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==========================
# LOAD SAVED OBJECTS
# ==========================
model = joblib.load("model/random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


# ==========================
# GUI WINDOW
# ==========================
root = tk.Tk()
root.title("Customer Churn Prediction System")
root.geometry("1400x900")


# ==========================
# TITLE
# ==========================
title_label = tk.Label(
    root,
    text="Customer Churn Prediction Dashboard",
    font=("Arial", 22, "bold")
)
title_label.pack(pady=10)


# ==========================
# MAIN FRAME
# ==========================
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)


# ==========================
# METRICS TEXT BOX
# ==========================
metrics_text = tk.Text(
    main_frame,
    height=10,
    font=("Consolas", 10)
)
metrics_text.pack(fill=tk.X, padx=10, pady=5)


# ==========================
# PLOT FRAME
# ==========================
plot_frame = tk.Frame(main_frame)
plot_frame.pack(fill=tk.BOTH, expand=True)


# ==========================
# PREPROCESS FUNCTION
# ==========================
def preprocess_data(df):
    # drop non-useful columns
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    # separate features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # categorical encoding
    X = pd.get_dummies(
        X,
        columns=['Geography', 'Gender'],
        drop_first=True
    )

    # align columns with training features
    X = X.reindex(columns=feature_columns, fill_value=0)

    # scaling
    X_scaled = scaler.transform(X)

    return X_scaled, y


# ==========================
# UPLOAD & RUN PIPELINE
# ==========================
def upload_and_run():
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
        return

    # preprocessing
    X, y = preprocess_data(df)

    # predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # metrics
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    # update metrics text
    metrics_text.delete("1.0", tk.END)
    metrics_text.insert(
        tk.END,
        f"Accuracy: {acc:.3f}\n\nClassification Report:\n{report}"
    )

    # clear old plots
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # ==========================
    # CREATE PLOTS
    # ==========================
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axs[0]
    )
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("Actual")

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    axs[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axs[1].plot([0, 1], [0, 1], linestyle="--")
    axs[1].set_title("ROC Curve")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend()

    # ---- Feature Importance ----
    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False).head(10)

    importance.plot(
        kind="barh",
        ax=axs[2]
    )
    axs[2].set_title("Top 10 Feature Importance")

    plt.tight_layout()

    # embed plots in tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# ==========================
# BUTTON
# ==========================
upload_btn = tk.Button(
    root,
    text="Upload CSV & Run Model",
    font=("Arial", 14),
    command=upload_and_run,
    bg="#2196F3",
    fg="white",
    padx=20,
    pady=10
)
upload_btn.pack(pady=10)


root.mainloop()
