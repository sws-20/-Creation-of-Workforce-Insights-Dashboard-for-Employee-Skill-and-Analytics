import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "encoded_data.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_weekly_agent():
    print("Agent starting weekly talent scanning...")
    
    # Check dependencies
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    model_path = os.path.join(MODELS_DIR, "lr_model.joblib")
    
    if not (os.path.exists(scaler_path) and os.path.exists(features_path) and os.path.exists(model_path)):
        raise FileNotFoundError("Models not trained yet. Run scripts/train.py first.")
        
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    model = joblib.load(model_path)
    
    # Load dataset
    df = pd.read_excel(DATA_PATH)
    
    # Reconstruct readable categorical features for reporting
    def get_categorical_value(row, prefix, default="Other"):
        cols = [c for c in df.columns if c.startswith(prefix + "_")]
        for c in cols:
            if row[c] == 1:
                return c.replace(prefix + "_", "")
        return default

    # Extract features for prediction
    X_raw = df[feature_names]
    X_scaled = scaler.transform(X_raw)
    
    # Predict probabilities
    probs = model.predict_proba(X_scaled)[:, 1]
    
    df['Predicted_Attrition_Prob'] = probs
    
    # Identify high risk employees (prob >= 0.50)
    high_risk_df = df[df['Predicted_Attrition_Prob'] >= 0.50].copy()
    high_risk_df = high_risk_df.sort_values(by='Predicted_Attrition_Prob', ascending=False)
    
    # Compile stats
    total_scanned = len(df)
    total_high_risk = len(high_risk_df)
    avg_health_score = df['Health_Score'].mean()
    total_skill_gaps = df['Skill_Gap_Flag'].sum()
    total_promotions = df['Promotion'].sum()
    
    # Draft markdown report
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_filename = "weekly_report_latest.md"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    report_content = f"""# Talent Intelligence Weekly Executive Report
**Report Generated:** {timestamp_str}
**Target Population Scanned:** {total_scanned} employees

---

## 📊 High-Level Workforce Summary
*   **Total High-Risk Attrition Flags:** {total_high_risk} ({total_high_risk / total_scanned * 100:.1f}%)
*   **Average Workforce Health Score:** {avg_health_score:.1f} / 100
*   **Total Active Skill Gaps:** {total_skill_gaps} employees flagged
*   **Total Promotion-Ready Candidates:** {total_promotions} employees flagged

---

## 🚨 Top At-Risk Employees Flagged for Review
Below are the top 5 highest-risk employees currently flagged by the predictive model, requiring immediate HR retention review:

| Employee ID | Department | Job Role | Attrition Prob | Health Score | Skill Gap? | Promotion Ready? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    
    # Take top 5 for listing
    top_at_risk = high_risk_df.head(5)
    for idx, row in top_at_risk.iterrows():
        emp_id = int(row['EmployeeNumber'])
        
        # Get reconstructed categoricals
        dept = get_categorical_value(row, "Department", "Research & Development")
        role = get_categorical_value(row, "JobRole", "Associate")
        
        prob = row['Predicted_Attrition_Prob'] * 100
        health = row['Health_Score']
        skill_gap = "Yes ⚠️" if row['Skill_Gap_Flag'] == 1 else "No"
        promo = "Yes 🚀" if row['Promotion'] == 1 else "No"
        
        report_content += f"| #{emp_id} | {dept} | {role} | {prob:.1f}% | {health:.1f} | {skill_gap} | {promo} |\n"
        
    report_content += f"""
---

## 💡 Recommended HR Action Items
Based on this week's predictive analysis, we recommend the following strategic steps:

1.  **Retention Outreach for High-Risk Staff:**
    *   Schedule 1-on-1 pulse checks for employees with attrition probability > 70%.
    *   Check for high workload/overtime triggers (particularly in Sales and Engineering departments).
2.  **Addressing Skill Gaps:**
    *   Ensure the {total_skill_gaps} employees flagged with **Skill Gaps** are enrolled in their required skill-building tracks.
    *   Pair them with mentors for performance acceleration.
3.  **Capitalizing on Promotion Readiness:**
    *   Review the {total_promotions} **Promotion-Ready** candidates in the upcoming bi-annual cycle.
    *   Recognizing top performance is one of our strongest retention mechanisms.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Weekly talent scanning completed! Executive report written to {report_path}")
    return report_path

if __name__ == "__main__":
    run_weekly_agent()
