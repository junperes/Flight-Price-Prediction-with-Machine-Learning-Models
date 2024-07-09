# Predicting Flight Prices: Exploring Data and Evaluating ML Algorithms 🛫💸
Explore our journey through flight price prediction using data analysis and machine learning. Compare algorithms to find the best predictor for your next trip's fare.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Technologies Used](#technologies-used)
- [Data Exploration](#data-exploration)
- [Model Comparison](#model-comparison)

## Overview
This project aims to predict flight prices using various machine learning algorithms. We begin by exploring and analyzing flight data to extract meaningful insights. Key steps include data preprocessing, feature engineering, and model evaluation using different algorithms such as Linear Regression and Random Forests.

## Requirements
To run this project locally, ensure you have Python (version 3.12.2) installed. Clone this repository and install the required libraries using pip:

`pip install -r requirements.txt`

## Technologies Used

- **Python Libraries:**
  - `pandas` for data manipulation and analysis.
  - `numpy` for numerical operations and array manipulation.
  - `seaborn` for statistical data visualization.
  - `matplotlib` for creating plots and charts.
  - `scipy.stats.randint` for generating random integers.

- **Machine Learning Libraries:**
  - `scikit-learn`:
    - `LinearRegression` for fitting a linear regression model.
    - `StandardScaler` for standardizing features.
    - `train_test_split` for splitting data into training and testing sets.
    - `RandomizedSearchCV` for hyperparameter tuning using random search.
    - `RandomForestRegressor` for fitting a random forest regression model.
    - Evaluation metrics: `mean_absolute_error`, `mean_squared_error`, `r2_score`.

## Data Exploration
In this section, we explore the dataset used for predicting flight prices. 

- The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).
- Overview of dataset features and their relevance to flight price prediction.
- Visualizations (e.g., histograms, scatter plots) highlighting key insights from the data.

## Model Comparison

| Model                | Mean Absolute Error (₹) | Mean Absolute Error (USD)   | R-squared Score |
|----------------------|-------------------------|-----------------------------|-----------------|
| Linear Regression    | ₹4499.0                 | ≈ $63                       | 0.91            |
| Random Forest        | ₹1076.4                 | ≈ $13                       | 0.98            |

