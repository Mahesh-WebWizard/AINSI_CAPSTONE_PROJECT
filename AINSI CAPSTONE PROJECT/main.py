# Project Title: Predicting Average Fuel Consumption using Artificial Neural Networks (ANN) in a GUI-based Application

# Import necessary libraries
from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import os

# Initialize main window
main = Tk()
main.title("Average Fuel Consumption")
main.geometry("1300x700")

# Global variables for use across functions
filename = ''
data = None
train_x, test_x, train_y, test_y = None, None, None, None
model = None

# Function to upload dataset from the local system
def upload():
    global filename, data
    filename = filedialog.askopenfilename(initialdir=".", title="Select Dataset", filetypes=[("CSV Files", "*.csv")])
    if filename:
        data = pd.read_csv(filename)
        text.insert(END, f"{filename} loaded\n")
        text.insert(END, str(data.head()) + "\n")

# Function to preprocess data and split into training and test sets
def generate_model():
    global data, train_x, test_x, train_y, test_y
    if data is None:
        text.insert(END, "No data loaded!\n")
        return

    try:
        X = data[['Engine Size', 'Fuel Type', 'Vehicle Weight', 'Number of Cylinders']]
        X = pd.get_dummies(X)  # One-hot encode categorical columns

        y = data['Fuel Consumption']
        y = pd.cut(y, bins=10, labels=False)

        encoder = OneHotEncoder(sparse_output=False)
        Y = encoder.fit_transform(y.values.reshape(-1, 1))

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
        text.insert(END, f"Train/Test split done. Train size: {len(train_x)}, Test size: {len(test_x)}\n")
    except Exception as e:
        text.insert(END, f"Error in generating model data: {str(e)}\n")

# Function to build and train the ANN
def train_ann():
    global model, train_x, train_y, test_x, test_y
    if train_x is None or train_y is None:
        text.insert(END, "Please generate train/test data first.\n")
        return

    try:
        model = Sequential()
        model.add(Dense(64, input_shape=(train_x.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(train_y.shape[1], activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=2)

        acc = model.evaluate(test_x, test_y, verbose=0)[1]
        text.insert(END, f"Model trained successfully. Test accuracy: {acc * 100:.2f}%\n")
    except Exception as e:
        text.insert(END, f"Training failed: {str(e)}\n")

# Function to predict new data
def predict():
    global model
    if model is None:
        text.insert(END, "Model is not trained yet.\n")
        return

    filename = filedialog.askopenfilename(initialdir=".", title="Select Test Dataset", filetypes=[("CSV Files", "*.csv")])
    if not filename:
        return

    try:
        test_data = pd.read_csv(filename)
        X_new = test_data[['Engine Size', 'Fuel Type', 'Vehicle Weight', 'Number of Cylinders']]
        X_new = pd.get_dummies(X_new)

        # Ensure same columns as training data
        missing_cols = set(train_x.columns) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0
        X_new = X_new[train_x.columns]

        predictions = model.predict(X_new)
        predicted_classes = predictions.argmax(axis=1)

        for i, pred in enumerate(predicted_classes):
            text.insert(END, f"Record {i + 1}: Predicted Fuel Class: {pred}\n")
    except Exception as e:
        text.insert(END, f"Prediction error: {str(e)}\n")

# Function to plot prediction graph
def plot_graph():
    if model is None or test_x is None:
        text.insert(END, "Model not trained or test data unavailable.\n")
        return

    try:
        y_pred = model.predict(test_x).argmax(axis=1)
        y_true = test_y.argmax(axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted', linestyle='dashed')
        plt.xlabel('Sample Index')
        plt.ylabel('Fuel Consumption Class')
        plt.title('Actual vs Predicted Fuel Consumption')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        text.insert(END, f"Plot error: {str(e)}\n")

# Function to generate synthetic data
def generate_synthetic_data():
    num_samples = 500
    engine_sizes = [round(random.uniform(1.0, 5.0), 1) for _ in range(num_samples)]
    fuel_types = [random.choice(['Petrol', 'Diesel']) for _ in range(num_samples)]
    vehicle_weights = [random.randint(800, 2500) for _ in range(num_samples)]
    cylinders = [random.choice([4, 6, 8]) for _ in range(num_samples)]

    fuel_consumptions = []
    for i in range(num_samples):
        base = 20 - (engine_sizes[i] * 2) - ((vehicle_weights[i] - 800) / 500) - (cylinders[i] * 0.5)
        if fuel_types[i] == 'Diesel':
            base += 2
        fuel_consumptions.append(round(max(5, min(25, base)), 1))

    df = pd.DataFrame({
        'Engine Size': engine_sizes,
        'Fuel Type': fuel_types,
        'Vehicle Weight': vehicle_weights,
        'Number of Cylinders': cylinders,
        'Fuel Consumption': fuel_consumptions
    })

    df.to_csv("synthetic_fuel_data.csv", index=False)
    text.insert(END, "Synthetic dataset saved as 'synthetic_fuel_data.csv'\n")

# GUI Components
title = Label(main, text='ML Model for Average Fuel Consumption', bg='greenyellow', fg='dodgerblue', font=('times', 16, 'bold'), height=2, width=100)
title.pack()

text = Text(main, height=20, width=100)
text.pack(pady=10)

frame = Frame(main)
frame.pack(pady=10)

Button(frame, text="Generate Synthetic Data", command=generate_synthetic_data, width=25).grid(row=0, column=0, padx=10)
Button(frame, text="Upload Dataset", command=upload, width=20).grid(row=0, column=1, padx=10)
Button(frame, text="Generate Train/Test Data", command=generate_model, width=25).grid(row=0, column=2, padx=10)
Button(frame, text="Train ANN", command=train_ann, width=15).grid(row=0, column=3, padx=10)
Button(frame, text="Predict Fuel Consumption", command=predict, width=25).grid(row=0, column=4, padx=10)
Button(frame, text="Plot Graph", command=plot_graph, width=15).grid(row=0, column=5, padx=10)
Button(frame, text="Exit", command=main.destroy, width=10).grid(row=0, column=6, padx=10)

main.config(bg='LightSkyBlue')
main.mainloop()
