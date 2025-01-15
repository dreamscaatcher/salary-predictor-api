# import packages for data manipulation
import pandas as pd
import numpy as np

# import packages for machine learning
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score

# import packages for data management
import joblib
import json

# Load and prepare the data
print("Loading data...")
salary_data = pd.read_csv('data/ds_salaries.csv')

# Store unique values before encoding
unique_values = {
    'experience_levels': sorted(salary_data['experience_level'].unique().tolist()),
    'company_sizes': sorted(salary_data['company_size'].unique().tolist()),
    'employment_types': sorted(salary_data['employment_type'].unique().tolist()),
    'job_titles': sorted(salary_data['job_title'].unique().tolist())
}

print("\nUnique values in each column:")
for key, values in unique_values.items():
    print(f"{key}:", values)

# Clean up the data
print("\nCleaning data...")
# Keep only relevant columns
keep_columns = ['experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'company_size']
salary_data = salary_data[keep_columns]

# Remove any rows with missing values
salary_data = salary_data.dropna()

# use ordinal encoder to encode experience level
print("\nEncoding experience level...")
encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
salary_data['experience_level_encoded'] = encoder.fit_transform(salary_data[['experience_level']])

# use ordinal encoder to encode company size
print("Encoding company size...")
encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
salary_data['company_size_encoded'] = encoder.fit_transform(salary_data[['company_size']])

# encode employment type and job title using dummy columns
print("Creating dummy variables...")
salary_data = pd.get_dummies(salary_data, columns=['employment_type', 'job_title'], drop_first=True, dtype=int)

# drop original columns
salary_data = salary_data.drop(columns=['experience_level', 'company_size'])

print("\nFinal columns for model:", salary_data.columns.tolist())

# define independent and dependent features
X = salary_data.drop(columns='salary_in_usd')
y = salary_data['salary_in_usd']

# Save feature information
feature_info = {
    'columns': X.columns.tolist(),
    'unique_values': unique_values
}

# Create model directory if it doesn't exist
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
print(f"\nCreating model directory at {MODEL_DIR}...")
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model directory exists: {os.path.exists(MODEL_DIR)}")

# Save feature names for reference
feature_info_path = os.path.join(MODEL_DIR, 'feature_info.json')
print(f"\nSaving feature info to {feature_info_path}...")
with open(feature_info_path, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"Feature info file exists: {os.path.exists(feature_info_path)}")
print("\nFeature info saved successfully")

# split between training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=104, test_size=0.2, shuffle=True)

print("\nTraining model...")
# fit linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# make predictions
y_pred = regr.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean squared error: {mse:.2f}")
print(f"R2 score: {r2:.2f}")

# save model using joblib
model_path = os.path.join(MODEL_DIR, 'lin_regress.sav')
print(f"\nSaving model to {model_path}...")
joblib.dump(regr, model_path)
print(f"Model file exists: {os.path.exists(model_path)}")
print("Model saved successfully!")

# Print final feature names for verification
print("\nFinal feature names:")
print(X.columns.tolist())
