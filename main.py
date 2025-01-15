from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import OrdinalEncoder
import joblib
import pandas as pd
import json

app = FastAPI()

# Get allowed origins from environment variable
import os
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class PredictionFeatures(BaseModel):
    experience_level: str
    company_size: str
    employment_type: str
    job_title: str

class PredictionResponse(BaseModel):
    salary: float

# Set up model directory path
import os
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# Load the model and feature info
model = joblib.load(os.path.join(MODEL_DIR, 'lin_regress.sav'))
with open(os.path.join(MODEL_DIR, 'feature_info.json'), 'r') as f:
    feature_info = json.load(f)

@app.get("/")
async def root():
    return {"message": "Welcome to the Salary Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PredictionFeatures):
    # Create a DataFrame with a single row
    input_df = pd.DataFrame([{
        'experience_level': features.experience_level,
        'company_size': features.company_size,
        'employment_type': features.employment_type,
        'job_title': features.job_title.replace('_', ' ')  # Convert underscores to spaces
    }])
    
    # Encode experience level
    exp_encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
    input_df['experience_level_encoded'] = exp_encoder.fit_transform(input_df[['experience_level']])
    
    # Encode company size
    size_encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
    input_df['company_size_encoded'] = size_encoder.fit_transform(input_df[['company_size']])
    
    # Create dummy variables for employment type
    emp_type_dummies = pd.get_dummies(input_df['employment_type'], prefix='employment_type')
    if 'employment_type_CT' not in emp_type_dummies.columns:
        emp_type_dummies['employment_type_CT'] = 0
    if 'employment_type_FL' not in emp_type_dummies.columns:
        emp_type_dummies['employment_type_FL'] = 0
    if 'employment_type_FT' not in emp_type_dummies.columns:
        emp_type_dummies['employment_type_FT'] = 0
    if 'employment_type_PT' not in emp_type_dummies.columns:
        emp_type_dummies['employment_type_PT'] = 0
    
    # Create dummy variables for job title with all possible values
    job_dummies = pd.DataFrame(0, index=input_df.index, columns=[f'job_title_{title}' for title in feature_info['unique_values']['job_titles']])
    current_job = f'job_title_{features.job_title.replace("_", " ")}'
    if current_job in job_dummies.columns:
        job_dummies[current_job] = 1
    
    # Combine all features
    input_df = pd.concat([
        input_df[['experience_level_encoded', 'company_size_encoded']],
        emp_type_dummies,
        job_dummies
    ], axis=1)
    
    # Ensure columns match training data
    expected_columns = feature_info['columns']
    missing_cols = set(expected_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[expected_columns]
    
    # Drop one category from employment type (as was done during training)
    if 'employment_type_CT' in input_df.columns:
        input_df = input_df.drop('employment_type_CT', axis=1)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return PredictionResponse(salary=prediction)
