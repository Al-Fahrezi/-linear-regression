Dataset from https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah?resource=download&select=HARGA+RUMAH+JAKSEL.xlsx

Project Description for GitHub: House Price Prediction Using Linear Regression with Regularization

Overview

This project aims to build a linear regression model that can predict house prices based on several features such as 

building area (LB), 
land area (LT), 
number of bedrooms (KT), 
number of bathrooms (KM), 
and number of garages (GRS). 

To improve the model's performance and prevent overfitting, regularization techniques are employed.

Project Steps
1. Data Loading:
Load the house data from an Excel file named DATA RUMAH.xlsx.
Feature Normalization:

2. Normalize the features to ensure they are on the same scale. This is achieved by calculating the mean and standard deviation of each feature and then transforming the features accordingly.

3. Add Bias Term: Add a bias column (with value 1) to the normalized features to allow the model to learn the intercept of the linear regression.

4. Cost Function with Regularization:The cost function computes the error (difference between predictions and actual values) and adds a regularization penalty to reduce model complexity and prevent overfitting.

5. Gradient Descent with Regularization: Use gradient descent to update the model weights. Regularization is added in the weight update process except for the bias term.

6. Data Preparation: Split the data into training and testing sets with an 80:20 ratio.

7. Hyperparameter Experimentation: Try various combinations of learning rates (0.001, 0.01, 0.1) and regularization strengths (0, 0.01, 0.1) to find the best combination that gives the lowest cost on the training set.

8. Prediction and Evaluation:Use the best model to predict house prices on the test set. Evaluate the predictions using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2).Visualize the results to compare actual and predicted prices, and to see the convergence of the cost function during gradient descent.

9. Final Results:Visualize a scatter plot showing the relationship between actual and predicted house prices. Plot the cost function convergence for the best model. Print the best weights, learning rate, and regularization strength found. Display the evaluation metrics: MAE, MSE, and R2.
