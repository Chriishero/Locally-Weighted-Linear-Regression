# Locally Weighted Linear Regression (from scratch)
Simple implementation of a locally weighted linear regression in python, without any external library (excluding NumPy).

## Description
This project implements a basic Locally Weighted Linear Regression predictor.
LWR is a non-parametric regression method: for each query point, the model computes a local set of weight and solves a weighted normal equation. This implementation can also use Gradient Descent, even if it is NEVER used with LWR. The only hyperparameters included are :
- n_iteration (>0)
- learning_rate (>0)
- tau (>0)
- method (gradient_descent or normal_equation)
- verbosity (0, 1 or 2)

## Usage
```bash
from Locally_Weighted_Linear_Regression.LWR import LocallyWeightedLinearRegression

predictor = LocallyWeightedLinearRegression.create_predictor()
y_pred = predictor.predict(X, y, x_query)
```