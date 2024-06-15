#Objective

The objective of this project is to build a linear regression model to predict house prices based on various features such as building area (LB), land area (LT), number of bedrooms (KT), number of bathrooms (KM), and number of garages (GRS). This project involves implementing all components of the model, including data normalization, cost function, and gradient descent algorithm, from scratch without using pre-existing machine learning libraries.

Dataset
The dataset used in this project contains information about houses, including the following features:

LB (Building Area): Building area in square meters.
LT (Land Area): Land area in square meters.
KT (Number of Bedrooms): Number of bedrooms.
KM (Number of Bathrooms): Number of bathrooms.
GRS (Number of Garages): Number of garages.
Price: House price in Indonesian Rupiah.
The dataset is loaded from an Excel file named DATA RUMAH.xlsx.

Project Steps
Import Necessary Libraries

Importing libraries such as numpy for numerical operations, pandas for data manipulation, and matplotlib for visualization.
Load Data from Excel File

Reading the dataset from the Excel file using pandas.
Normalize Features

Normalizing the features to ensure they are on the same scale. This is done by subtracting the mean and dividing by the standard deviation of each feature.
Add Bias Column

Adding a column of ones at the beginning of the feature matrix to account for the bias term in the linear regression model.
Create Cost Function with Regularization

Implementing a cost function that calculates the Mean Squared Error (MSE) and adds a regularization term to reduce overfitting.
Gradient Descent with Regularization

Implementing the gradient descent algorithm to minimize the cost function. A regularization term is added to penalize large feature weights and reduce overfitting.
Prepare Data

Splitting the dataset into features (X) and target (y).
Normalizing the features and adding a bias column.
Splitting the data into training and testing sets.
Experiment with Different Learning Rates and Regularization Values

Trying different combinations of learning rates and regularization values to find the best combination that results in the lowest cost.
Make Predictions on Test Data

Using the best model to predict house prices on the test data.
Visualize Results

Creating a plot to compare actual prices with predicted prices to evaluate the model's performance.
Visualize Cost Function Convergence

Creating a plot to show the convergence of the cost function during the training process to ensure the gradient descent algorithm is working correctly.
Print Best Weights and Hyperparameters

Displaying the best weights found, along with the learning rate and regularization value used.
