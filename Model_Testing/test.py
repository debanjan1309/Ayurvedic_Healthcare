import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv("training_dataset.csv")
df_test = pd.read_csv("testing_dataset.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

# Extract symptoms and prognosis columns
symptoms = df.columns[:-1]
prognosis = df.columns[-1]

# Replace prognosis with integer values for training
prognosis_dict = {disease: i for i, disease in enumerate(df[prognosis].unique())}
df[prognosis] = df[prognosis].map(prognosis_dict)
df_test[prognosis] = df_test[prognosis].map(prognosis_dict)

# Prepare feature and target variables
X = df[symptoms]
y = df[prognosis]
X_test = df_test[symptoms]
y_test = df_test[prognosis]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train models and check for overfitting
for name, model in models.items():
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    print(f"{name} Training Accuracy: {train_accuracy:.2f}")
    print(f"{name} Validation Accuracy: {val_accuracy:.2f}")

# Perform cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# Function to predict the disease based on user input symptoms
def predict_disease(symptoms_input, models, X_test, y_test):
    input_vector = [0] * len(symptoms)
    for symptom in symptoms_input:
        if symptom in symptoms:
            input_vector[symptoms.get_loc(symptom)] = 1
    
    results = {}
    for name, model in models.items():
        prediction = model.predict([input_vector])[0]
        disease = list(prognosis_dict.keys())[list(prognosis_dict.values()).index(prediction)]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = (disease, accuracy)
    
    return results

# Tkinter GUI for symptom input and prediction
def show_prediction():
    symptoms_input = [symptom_entry.get()]
    results = predict_disease(symptoms_input, models, X_test, y_test)
    result_text = ""
    for name, (disease, accuracy) in results.items():
        result_text += f"{name} - Predicted Disease: {disease}, Accuracy: {accuracy:.2f}\n"
    messagebox.showinfo("Prediction Results", result_text)

root = tk.Tk()
root.title("Disease Prediction")

tk.Label(root, text="Enter Symptom:").pack()
symptom_entry = tk.Entry(root)
symptom_entry.pack()

tk.Button(root, text="Predict Disease", command=show_prediction).pack()

root.mainloop()
