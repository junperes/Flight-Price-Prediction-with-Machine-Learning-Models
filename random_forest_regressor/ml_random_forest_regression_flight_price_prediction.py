import pandas as pd
import math
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import randint

# - Loading, Inspecting, and Treating Data

df = pd.read_csv('Clean_Dataset.csv') # Dataset from kaggle

print(df) # Display the first few rows of the dataframe
print(df.info()) # Display the summary information about the dataframe

df = df.drop(columns=['Unnamed: 0', 'flight']) # Drop unnecessary columns

# Check and handle missing values
print(df.isnull().sum()) # (0 missing values observed)

for column in df.columns: # Count unique values in each column
    unique_count = df[column].nunique()
    print(f"'{column}' has {unique_count} unique values.")

# - Transform the variables to numerical format:

# Convert 'class' column to binary format (2 distinct values)
df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0) 

# Encode 'stops' column as numeric labels (3 distinct values)
df['stops'] = pd.factorize(df['stops'])[0]

# Encode categorical columns as dummy variables (More than 3 distinct values)
columns_to_encode = ['airline', 'source_city', 'destination_city', 'departure_time', 'arrival_time']

for column in columns_to_encode:
    df = df.join(pd.get_dummies(df[column], prefix=column)).drop(column, axis=1)

# Check for Outliers
# plt.figure(figsize=(8, 6))
# sns.boxplot(x=df['price'])
# plt.title('Box plot of Price')
# plt.show()

# As there are outliers in the price data, normally we would treat them for linear regression. 
# However, to compare models effectively, I will proceed without treating them.

X, y = df.drop('price', axis=1), df['price'] # Assign features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split data into training and testing sets

scaler = StandardScaler() #To ensure that all features contribute equally to the model fitting process. 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# - Random Forest Regressor 

reg= RandomForestRegressor(n_jobs=-1)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print('R2: ', r2_score(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', math.sqrt(mean_squared_error(y_test, y_pred)))

# To visualize: scatter plot comparing actual flight prices versus predicted prices by the model.
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Flight Price')
# plt.ylabel('Predicted Flight Price')
# plt.title('Prediction VS Actual Price')
# plt.show()

# Calculate feature importances and sort them in descending order
importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print(sorted_importances)
# Note: The class is the most important by far

# plt.figure(figsize=(15,6))
# plt.bar([x[0] for x in sorted_importances[:5]], [x[1] for x in sorted_importances[:5]])
# plt.show()

# Due to the exhaustive nature of GridSearchCV, which can be time-consuming when searching through a large parameter grid, an alternative approach is RandomizedSearchCV.
# Once we identify promising parameter ranges using RandomizedSearchCV, we can then refine our search using GridSearchCV if necessary

# Hyperparameter Tune

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
}

reg = RandomForestRegressor(n_jobs=-1)

random_search = RandomizedSearchCV(estimator=reg, param_distributions=param_dist, n_iter=2, cv=3,
                                   scoring='neg_mean_squared_error', verbose=2, random_state=10, n_jobs=-1)

random_search.fit(X_train_scaled, y_train)

best_regressor = random_search.best_estimator_

print(best_regressor)

y_pred = best_regressor.predict(X_test_scaled)
print('R2: ', r2_score(y_test, y_pred))
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))
print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', math.sqrt(mean_squared_error(y_test, y_pred)))

# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Flight Price')
# plt.ylabel('Predicted Flight Price')
# plt.title('Prediction VS Actual Price')
# plt.show()

# R2:  0.9849616692548722
# MAE:  1077.4469591255438
# MSE:  7751992.923179185
# RMSE:  2784.2400979763192

# to:
# R2:  0.985912663093094
# Mean absolute error:  1090.078059090289
# Mean squared error:  7261772.457302621
# RMSE:  2694.7676072905842
# just a little bit better

# Others: XGBoost/LightGBM(Gradient Boosting Regressor), SVR(Support Vector Regressor), MPL (Redes Neurais)
