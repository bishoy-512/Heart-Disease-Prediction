import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import messagebox

# Load and preprocess data
def load_and_prepare_data():
    df = pd.read_csv("D:\AI Project\project_ai\heart.csv")
    df['chol'] = pd.to_numeric(df['chol'], errors='coerce')
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    X = df.drop('target', axis=1)
    y = df['target']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, scaler, X.columns.tolist(), train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train all models and store them
def train_all_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf'),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

# Load data and train models
X, y, scaler, all_columns, (X_train, X_test, y_train, y_test) = load_and_prepare_data()
models = train_all_models(X_train, y_train)

# GUI Setup
root = tk.Tk()
root.title("Heart Disease Predictor")
root.geometry("900x900")

entries = {}
for idx, col_name in enumerate(all_columns):
    label = tk.Label(root, text=col_name.capitalize())
    label.grid(row=idx, column=0, padx=10, pady=5, sticky='e')
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[col_name] = entry

model_label = tk.Label(root, text="Select Model")
model_label.grid(row=len(all_columns), column=0, pady=10)
selected_model = tk.StringVar()
model_combo = ttk.Combobox(root, textvariable=selected_model, state="readonly")
model_combo['values'] = list(models.keys())
model_combo.current(0)
model_combo.grid(row=len(all_columns), column=1)

# Predict button function
def predict():
    try:
        input_data = [float(entries[col].get()) for col in all_columns]
        input_np = np.array([input_data])
        input_scaled = scaler.transform(input_np)

        model = models[selected_model.get()]
        prediction = model.predict(input_scaled)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

predict_btn = tk.Button(root, text="Predict", command=predict)
predict_btn.grid(row=len(all_columns)+1, column=0, columnspan=2, pady=20)

root.mainloop()