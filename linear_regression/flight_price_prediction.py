import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# - Linear Regression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print('Linear Regression:')

# R-squared (coefficient of determination) score measures how well the model fits the data. 
# A score closer to 1 indicates a better fit.
print('R2: ', r2_score(y_test, y_pred))

# Mean Absolute Error (MAE) measures the average magnitude of errors in predictions. 
# Lower values indicate better prediction accuracy.
print('MAE: ', mean_absolute_error(y_test, y_pred))

# Mean Squared Error (MSE) measures the average of the squared differences between predicted and actual values. 
# It gives a rough idea of the magnitude of error the model makes.
print('MSE: ', mean_squared_error(y_test, y_pred)) 
# MSE is sensitive to outliers, and a high value (as observed here) suggests the model may struggle to accurately predict extreme prices.

# Root Mean Squared Error (RMSE) is the square root of the MSE, providing a measure of how well the model predicts the response variable.
# Lower values indicate better fit between predicted and observed values.
print('RMSE: ', math.sqrt(mean_squared_error(y_test, y_pred))) 

# The model shows reasonable performance with good accuracy. 
# Assuming the currency is in Indian Rupees, the MAE of approximately $54 indicates a relatively small average prediction error. 
# To use this model effectively, addressing outliers would be advisable to further improve its reliability.
