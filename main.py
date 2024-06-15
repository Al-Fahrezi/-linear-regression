import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = 'DATA RUMAH.xlsx'
data = pd.read_excel(file_path)

# Normalize the features
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# Adding a column of ones to the features for the bias term
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Compute the cost function with regularization
def compute_cost(X, y, weights, regularization):
    m = len(y)
    predictions = X @ weights
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    reg_term = (regularization / (2 * m)) * np.sum(weights[1:] ** 2)
    return cost + reg_term

# Gradient descent algorithm with regularization
def gradient_descent(X, y, weights, learning_rate, iterations, regularization):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X @ weights
        errors = predictions - y
        gradients = (1 / m) * X.T @ errors
        regularization_term = (regularization / m) * weights
        regularization_term[0] = 0  # Don't regularize the bias term
        weights -= learning_rate * (gradients + regularization_term)
        cost = compute_cost(X, y, weights, regularization)
        cost_history.append(cost)

    return weights, cost_history

# Prepare the data
X = data[['LB', 'LT', 'KT', 'KM', 'GRS']].values
y = data['HARGA'].values

# Normalize the features
X_normalized, mean, std = normalize_features(X)
X_normalized = add_bias(X_normalized)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Experiment with different learning rates and regularization strengths
learning_rates = [0.001, 0.01, 0.1]
regularizations = [0, 0.01, 0.1]

best_weights = None
best_cost_history = None
best_learning_rate = None
best_regularization = None
lowest_cost = float('inf')

for lr in learning_rates:
    for reg in regularizations:
        weights = np.zeros(X_train.shape[1])
        weights, cost_history = gradient_descent(X_train, y_train, weights, lr, 1000, reg)
        final_cost = cost_history[-1]
        
        if final_cost < lowest_cost:
            lowest_cost = final_cost
            best_weights = weights
            best_cost_history = cost_history
            best_learning_rate = lr
            best_regularization = reg

# Make predictions on the test set
y_pred = X_test @ best_weights

# Visualizing the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Optimized Linear Regression Model - Predicted vs Actual Prices\n(LR={best_learning_rate}, Reg={best_regularization})')
plt.legend()
plt.show()

# Visualizing cost function convergence for the best model
plt.figure(figsize=(10, 6))
plt.plot(range(1000), best_cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence for Best Model')
plt.show()

# Print the best weights and hyperparameters
print(f'Best Weights: {best_weights}')
print(f'Best Learning Rate: {best_learning_rate}')
print(f'Best Regularization: {best_regularization}')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
