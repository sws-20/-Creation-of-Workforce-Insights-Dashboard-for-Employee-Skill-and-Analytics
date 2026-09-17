import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Workforce Analytics & Talent Intelligence API",
    description="Backend API exposing predictive attrition models, health score computations, and RAG HR query engine.",
    version="1.0.0"
)

# Allow the Front-End (running on a different origin/port) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models and scaler
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    print("Models and standardizer loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}. Make sure to run scripts/train.py first.")
    scaler, feature_names, lr_model, rf_model, xgb_model = None, None, None, None, None

# Define baseline default values for all 57 features (using typical/median values)
DEFAULT_EMPLOYEE_FEATURES = {
    'Age': 35.0, 'DailyRate': 800.0, 'DistanceFromHome': 9.0, 'Education': 3.0,
    'EmployeeCount': 1.0, 'EmployeeNumber': 1000.0, 'EnvironmentSatisfaction': 3.0,
    'Gender': 1, 'HourlyRate': 65.0, 'JobInvolvement': 3.0, 'JobLevel': 2.0,
    'JobSatisfaction': 3.0, 'MonthlyIncome': 5000.0, 'MonthlyRate': 14000.0,
    'NumCompaniesWorked': 2.0, 'Over18': 1, 'OverTime': 0, 'PercentSalaryHike': 15.0,
    'PerformanceRating': 3.0, 'RelationshipSatisfaction': 3.0, 'StandardHours': 80.0,
    'StockOptionLevel': 1.0, 'TotalWorkingYears': 10.0, 'TrainingTimesLastYear': 2.0,
    'WorkLifeBalance': 3.0, 'YearsAtCompany': 5.0, 'YearsInCurrentRole': 3.0,
    'YearsSinceLastPromotion': 1.0, 'YearsWithCurrManager': 3.0,
    'BusinessTravel_Travel_Frequently': 0, 'BusinessTravel_Travel_Rarely': 1,
    'Department_Research & Development': 1, 'Department_Sales': 0,
    'EducationField_Life Sciences': 1, 'EducationField_Marketing': 0,
    'EducationField_Medical': 0, 'EducationField_Other': 0,
    'EducationField_Technical Degree': 0, 'JobRole_Human Resources': 0,
    'JobRole_Laboratory Technician': 0, 'JobRole_Manager': 0,
    'JobRole_Manufacturing Director': 0, 'JobRole_Research Director': 0,
    'JobRole_Research Scientist': 0, 'JobRole_Sales Executive': 0,
    'JobRole_Sales Representative': 0, 'MaritalStatus_Married': 1,
    'MaritalStatus_Single': 0, 'Age_Group_Senior': 0, 'Age_Group_Young Adult': 0,
    'Experience_Level_Junior': 0, 'Experience_Level_Mid-Level': 1,
    'Experience_Level_Senior': 0, 'Income_Category_Low Income': 0,
    'Income_Category_Medium Income': 1, 'Tenure_Group_Long-Term Employee': 0,
    'Tenure_Group_New Employee': 0
}

class HealthScoreInput(BaseModel):
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction rating (1-4)")
    TrainingTimesLastYear: int = Field(..., ge=0, le=6, description="Training times in the last year (0-6)")
    PerformanceRating: int = Field(..., ge=1, le=4, description="Performance rating (1-4)")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Work-life balance rating (1-4)")

class EmployeeInput(BaseModel):
    # Optional values so the user can pass raw profiles, falling back to defaults for others
    Age: Optional[float] = None
    DailyRate: Optional[float] = None
    DistanceFromHome: Optional[float] = None
    Education: Optional[float] = None
    EnvironmentSatisfaction: Optional[float] = None
    Gender: Optional[str] = None  # "Male" or "Female"
    HourlyRate: Optional[float] = None
    JobInvolvement: Optional[float] = None
    JobLevel: Optional[float] = None
    JobSatisfaction: Optional[float] = None
    MonthlyIncome: Optional[float] = None
    MonthlyRate: Optional[float] = None
    NumCompaniesWorked: Optional[float] = None
    OverTime: Optional[str] = None  # "Yes" or "No"
    PercentSalaryHike: Optional[float] = None
    PerformanceRating: Optional[float] = None
    RelationshipSatisfaction: Optional[float] = None
    StockOptionLevel: Optional[float] = None
    TotalWorkingYears: Optional[float] = None
    TrainingTimesLastYear: Optional[float] = None
    WorkLifeBalance: Optional[float] = None
    YearsAtCompany: Optional[float] = None
    YearsInCurrentRole: Optional[float] = None
    YearsSinceLastPromotion: Optional[float] = None
    YearsWithCurrManager: Optional[float] = None
    # Categoricals (raw strings to be encoded by the API)
    BusinessTravel: Optional[str] = None  # "Travel_Frequently", "Travel_Rarely", "Non-Travel"
    Department: Optional[str] = None  # "Research & Development", "Sales", "Human Resources"
    EducationField: Optional[str] = None  # "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"
    JobRole: Optional[str] = None  # "Sales Executive", "Research Scientist", "Laboratory Technician", etc.
    MaritalStatus: Optional[str] = None  # "Single", "Married", "Divorced"
    # Or raw numerical feature dictionary can be passed
    RawFeatures: Optional[Dict[str, float]] = None

def process_raw_profile(emp: EmployeeInput) -> Dict[str, float]:
    """Helper to convert EmployeeInput schema to 57 numerical features."""
    if emp.RawFeatures:
        # Use provided features, fill in any missing from defaults
        features = DEFAULT_EMPLOYEE_FEATURES.copy()
        for k, v in emp.RawFeatures.items():
            if k in features:
                features[k] = v
        return features
        
    features = DEFAULT_EMPLOYEE_FEATURES.copy()
    
    # 1. Direct maps
    if emp.Age is not None: features['Age'] = emp.Age
    if emp.DailyRate is not None: features['DailyRate'] = emp.DailyRate
    if emp.DistanceFromHome is not None: features['DistanceFromHome'] = emp.DistanceFromHome
    if emp.Education is not None: features['Education'] = emp.Education
    if emp.EnvironmentSatisfaction is not None: features['EnvironmentSatisfaction'] = emp.EnvironmentSatisfaction
    if emp.HourlyRate is not None: features['HourlyRate'] = emp.HourlyRate
    if emp.JobInvolvement is not None: features['JobInvolvement'] = emp.JobInvolvement
    if emp.JobLevel is not None: features['JobLevel'] = emp.JobLevel
    if emp.JobSatisfaction is not None: features['JobSatisfaction'] = emp.JobSatisfaction
    if emp.MonthlyIncome is not None: features['MonthlyIncome'] = emp.MonthlyIncome
    if emp.MonthlyRate is not None: features['MonthlyRate'] = emp.MonthlyRate
    if emp.NumCompaniesWorked is not None: features['NumCompaniesWorked'] = emp.NumCompaniesWorked
    if emp.PercentSalaryHike is not None: features['PercentSalaryHike'] = emp.PercentSalaryHike
    if emp.PerformanceRating is not None: features['PerformanceRating'] = emp.PerformanceRating
    if emp.RelationshipSatisfaction is not None: features['RelationshipSatisfaction'] = emp.RelationshipSatisfaction
    if emp.StockOptionLevel is not None: features['StockOptionLevel'] = emp.StockOptionLevel
    if emp.TotalWorkingYears is not None: features['TotalWorkingYears'] = emp.TotalWorkingYears
    if emp.TrainingTimesLastYear is not None: features['TrainingTimesLastYear'] = emp.TrainingTimesLastYear
    if emp.WorkLifeBalance is not None: features['WorkLifeBalance'] = emp.WorkLifeBalance
    if emp.YearsAtCompany is not None: features['YearsAtCompany'] = emp.YearsAtCompany
    if emp.YearsInCurrentRole is not None: features['YearsInCurrentRole'] = emp.YearsInCurrentRole
    if emp.YearsSinceLastPromotion is not None: features['YearsSinceLastPromotion'] = emp.YearsSinceLastPromotion
    if emp.YearsWithCurrManager is not None: features['YearsWithCurrManager'] = emp.YearsWithCurrManager

    # String maps
    if emp.Gender is not None:
        features['Gender'] = 1 if emp.Gender.strip().lower() == "male" else 0
    if emp.OverTime is not None:
        features['OverTime'] = 1 if emp.OverTime.strip().lower() == "yes" else 0

    # Multi categorical variables
    if emp.BusinessTravel is not None:
        travel = emp.BusinessTravel.strip()
        features['BusinessTravel_Travel_Frequently'] = 1 if travel == "Travel_Frequently" else 0
        features['BusinessTravel_Travel_Rarely'] = 1 if travel == "Travel_Rarely" else 0

    if emp.Department is not None:
        dept = emp.Department.strip()
        features['Department_Research & Development'] = 1 if dept == "Research & Development" else 0
        features['Department_Sales'] = 1 if dept == "Sales" else 0

    if emp.EducationField is not None:
        field = emp.EducationField.strip()
        features['EducationField_Life Sciences'] = 1 if field == "Life Sciences" else 0
        features['EducationField_Marketing'] = 1 if field == "Marketing" else 0
        features['EducationField_Medical'] = 1 if field == "Medical" else 0
        features['EducationField_Other'] = 1 if field == "Other" else 0
        features['EducationField_Technical Degree'] = 1 if field == "Technical Degree" else 0

    if emp.JobRole is not None:
        role = emp.JobRole.strip()
        for r in ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 
                  'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']:
            features[f'JobRole_{r}'] = 1 if role == r else 0

    if emp.MaritalStatus is not None:
        status = emp.MaritalStatus.strip()
        features['MaritalStatus_Married'] = 1 if status == "Married" else 0
        features['MaritalStatus_Single'] = 1 if status == "Single" else 0

    # Dynamic Age / Tenure groups if missing
    age = features['Age']
    features['Age_Group_Senior'] = 1 if age >= 50 else 0
    features['Age_Group_Young Adult'] = 1 if age <= 30 else 0
    
    tenure = features['YearsAtCompany']
    features['Tenure_Group_Long-Term Employee'] = 1 if tenure >= 10 else 0
    features['Tenure_Group_New Employee'] = 1 if tenure <= 2 else 0

    income = features['MonthlyIncome']
    features['Income_Category_Low Income'] = 1 if income < 3000 else 0
    features['Income_Category_Medium Income'] = 1 if 3000 <= income <= 8000 else 0

    working_years = features['TotalWorkingYears']
    features['Experience_Level_Junior'] = 1 if working_years <= 3 else 0
    features['Experience_Level_Mid-Level'] = 1 if 3 < working_years <= 10 else 0
    features['Experience_Level_Senior'] = 1 if working_years > 10 else 0

    return features

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Workforce Analytics API is active. Exposing endpoints for attrition prediction, health score, and RAG HR policies."
    }

@app.post("/health-score")
def get_health_score(data: HealthScoreInput):
    weights = {
        'JobSatisfaction': 0.30,
        'TrainingTimesLastYear': 0.20,
        'PerformanceRating': 0.25,
        'WorkLifeBalance': 0.25,
    }
    training_norm = 1 + (data.TrainingTimesLastYear / 6.0) * 3.0
    raw = (
        weights['JobSatisfaction'] * data.JobSatisfaction +
        weights['TrainingTimesLastYear'] * training_norm +
        weights['PerformanceRating'] * data.PerformanceRating +
        weights['WorkLifeBalance'] * data.WorkLifeBalance
    )
    raw = max(1.0, min(4.0, raw))
    score = ((raw - 1.0) / 3.0) * 100
    
    # Categorize
    if score >= 75:
        category = "Healthy"
    elif score >= 50:
        category = "Moderate"
    else:
        category = "At-Risk"
        
    return {
        "health_score_raw": raw,
        "health_score": round(score, 2),
        "category": category
    }

@app.post("/predict-attrition")
def predict_attrition(employees: List[EmployeeInput]):
    if not lr_model or not scaler:
        raise HTTPException(status_code=500, detail="Models are not loaded on server. Run scripts/train.py.")
        
    results = []
    for idx, emp in enumerate(employees):
        # Preprocess features
        features_dict = process_raw_profile(emp)
        
        # Convert to list ordered by standard feature names
        ordered_features = [features_dict[name] for name in feature_names]
        
        # Convert to DataFrame with correct feature names
        X_df = pd.DataFrame([ordered_features], columns=feature_names)
        X_scaled = scaler.transform(X_df)
        
        # Predict using Logistic Regression (highest recall)
        prob = float(lr_model.predict_proba(X_scaled)[0, 1])
        prediction = int(lr_model.predict(X_scaled)[0])
        
        # Also predict with XGBoost as reference
        xgb_prob = float(xgb_model.predict_proba(X_df)[0, 1])
        
        # Calculate feature contributions using Logistic Regression coefficients
        # coef * X_scaled tells us how much each feature contributed to the log odds
        coef = lr_model.coef_[0]
        contributions = coef * X_scaled[0]
        
        # Sort contributions
        sorted_indices = np.argsort(contributions)
        
        # Top 3 drivers for leaving (largest positive log-odds contribution)
        top_risk_features = []
        for idx_contrib in reversed(sorted_indices[-3:]):
            name = feature_names[idx_contrib]
            val = float(contributions[idx_contrib])
            if val > 0:
                top_risk_features.append({"feature": name, "contribution": round(val, 4)})
                
        # Top 3 drivers for staying (largest negative log-odds contribution)
        top_retention_features = []
        for idx_contrib in sorted_indices[:3]:
            name = feature_names[idx_contrib]
            val = float(contributions[idx_contrib])
            if val < 0:
                top_retention_features.append({"feature": name, "contribution": round(val, 4)})
        
        results.append({
            "employee_index": idx,
            "attrition_prediction": prediction,
            "attrition_probability": round(prob, 4),
            "xgb_probability_ref": round(xgb_prob, 4),
            "risk_score_category": "High Risk" if prob >= 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk"),
            "top_risk_drivers": top_risk_features,
            "top_retention_drivers": top_retention_features
        })
        
    return {"predictions": results}

# Step 7 will inject the RAG query engine endpoint /query here.
try:
    from app.rag import SimpleRAG
except ImportError:
    from rag import SimpleRAG

class QueryInput(BaseModel):
    query: str = Field(..., description="The HR policy question to ask.")

try:
    rag_engine = SimpleRAG()
    print("RAG query engine initialized successfully.")
except Exception as e:
    print(f"Error initializing RAG engine: {e}")
    rag_engine = None

@app.post("/query")
def query_hr_policies(payload: QueryInput):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG query engine is not initialized.")
    return rag_engine.query(payload.query)

