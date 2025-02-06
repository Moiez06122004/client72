# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Settings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv('IEA Global EV Data 2024.csv')
data.dropna(inplace=True)  # Drop rows with any null values

# Set target column name to 'value'
target_column = 'value'
if target_column not in data.columns:
    raise ValueError(f"The specified target column '{target_column}' does not exist in the dataset.")

# Encode categorical variables (basic encoding with integers)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = pd.factorize(data[col])[0]

# Separate features (X) and target (y)
X = data.drop(columns=[target_column]).values  # Features as numpy array
y = data[target_column].values  # Target variable as numpy array

# Split the data into training and testing sets (70% train, 30% test)
split_index = int(0.7 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Add an intercept term (column of 1's) to the feature set for bias in linear regression
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Define the range of epochs
epochs = list(range(10, min(210, len(X_train)), 10))  # Step by 10, up to 200 or length of training data

# Lists to store R² and MAE values
train_r2_scores, test_r2_scores, train_maes, test_maes = [], [], [], []

# Train and evaluate at each epoch
for epoch in epochs:
    # Use a portion of training data for each "epoch"
    partial_X_train = X_train[:epoch]
    partial_y_train = y_train[:epoch]

    # Calculate theta (parameters) using the normal equation
    theta = np.linalg.inv(partial_X_train.T.dot(partial_X_train)).dot(partial_X_train.T).dot(partial_y_train)

    # Predictions on the training and test sets
    y_train_pred = partial_X_train.dot(theta)
    y_test_pred = X_test.dot(theta)

    # Calculate R² scores
    train_r2 = 1 - np.sum((partial_y_train - y_train_pred) ** 2) / np.sum((partial_y_train - np.mean(partial_y_train)) ** 2)
    test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    # Calculate MAE
    train_mae = np.mean(np.abs(partial_y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))

    # Store metrics
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    train_maes.append(train_mae)
    test_maes.append(test_mae)

    # Print metrics for each epoch
    print(f"Epoch {epoch}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

# Final Epoch Metrics Summary
print("\nFinal Epoch Metrics")
print("Train R² Scores:", train_r2_scores[-1])
print("Test R² Scores:", test_r2_scores[-1])
print("Train MAE:", train_maes[-1])
print("Test MAE:", test_maes[-1])

# Plot MAE and R² over epochs
plt.figure(figsize=(12, 6))

# Plot Training and Testing R² Score over Epochs
plt.subplot(1, 2, 1)
plt.plot(epochs, train_r2_scores, label="Train R² Score", marker='o')
plt.plot(epochs, test_r2_scores, label="Test R² Score", marker='o')
plt.xlabel("Epochs (Portion of Training Data)")
plt.ylabel("R² Score")
plt.title("Training vs Testing R² Score over Epochs")
plt.legend()

# Plot Training and Testing MAE over Epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, train_maes, label="Train MAE", marker='o')
plt.plot(epochs, test_maes, label="Test MAE", marker='o')
plt.xlabel("Epochs (Portion of Training Data)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Training vs Testing MAE over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
