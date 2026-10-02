from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import numpy as np
import os

app = FastAPI()

# Allow React frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "general": joblib.load("models/clf_general.joblib"),
    "california": joblib.load("models/clf_HeadquartersState_California.joblib"),
    "illinois": joblib.load("models/clf_HeadquartersState_Illinois.joblib"),
    "michigan": joblib.load("models/clf_HeadquartersState_Michigan.joblib"),
    "newyork": joblib.load("models/clf_HeadquartersState_New York.joblib"),
    "pennsylvania": joblib.load("models/clf_HeadquartersState_Pennsylvania.joblib"),
    "texas": joblib.load("models/clf_HeadquartersState_Texas.joblib"),
    "business_services": joblib.load("models/clf_Sector_Business Services.joblib"),
    "financials": joblib.load("models/clf_Sector_Financials.joblib"),
    "health_care": joblib.load("models/clf_Sector_Health Care.joblib"),
    "tech": joblib.load("models/clf_Sector_Technology.joblib"),
    "specialty_retailers": joblib.load("models/clf_Industry_Specialty_Retailers_Other.joblib"),
    "financials": joblib.load("models/clf_Sector_Financials.joblib")
}

class StockInput(BaseModel):
    Ticker: Optional[str] = None
    EPS_est: Optional[float] = None
    TTM_EPS: Optional[float] = None
    PE_Ratio: Optional[float] = None
    Forward_PE: Optional[float] = None
    Open: Optional[float] = None
    High: Optional[float] = None
    Low: Optional[float] = None
    Volume: Optional[float] = None
    daily_change: Optional[float] = None
    w1_change: Optional[float] = None
    m1_change: Optional[float] = None
    m3_change: Optional[float] = None
    y1_change: Optional[float] = None
    wk52_high: Optional[float] = None
    wk52_low: Optional[float] = None
    sma20_change: Optional[float] = None
    sma50_change: Optional[float] = None
    sma200_change: Optional[float] = None
    rsi_14: Optional[float] = None
    volatility_wk: Optional[float] = None
    volatility_1m: Optional[float] = None
    shifted_rep_eps: Optional[float] = None
    surprise_shift1: Optional[float] = None
    high_low_range: Optional[float] = None
    vol_log: Optional[float] = None
    Quarter: Optional[int] = None
    Sector: Optional[str] = None
    Industry: Optional[str] = None
    HeadquartersState: Optional[str] = None
    # open_sma20_ratio: Optional[float] = None

LOG_FILE = 'data/user_inputs.csv'

num_feats = ['EPS_est', 'TTM_EPS', 'PE_Ratio', 'Forward_PE', 'Open',
             'High', 'Low', 'Volume', 'daily_change', 'w1_change', 'm1_change', 'm3_change',
             'y1_change', 'wk52_high', 'wk52_low', 'sma20_change', 'sma50_change',
             'sma200_change', 'rsi_14', 'volatility_wk', 'volatility_1m', 
             'shifted_rep_eps', 'surprise_shift1', 'high_low_range', 'vol_log']

cat_feats = ['Quarter', 'Sector', 'Industry', 'HeadquartersState']

@app.post("/predict")
def predict(input: StockInput):
    data_dict = input.model_dump()
    
    # COLUMN_MAP = {
    #     "EPS_Estimate": "EPS Estimate",
    #     "w1_change": "1w_change",
    #     "m1_change": "1m_change",
    #     "m3_change": "3m_change",
    #     "y1_change": "1yr_change",
    #     "wk52_high": "52wk_high",
    #     "wk52_low": "52wk_low"
    # }

    # Replace None with np.nan (for numeric) or "MISSING" (for categorical)
    processed = {}
    for col, val in data_dict.items():
        if val is None:
            if col in cat_feats:  
                processed[col] = "MISSING"
            else:
                processed[col] = np.nan
        else:
            processed[col] = val
    
    df_log = pd.DataFrame([processed])
    df = df_log[num_feats + cat_feats]
    print("Input data:", processed)

    probs = [models["general"].predict_proba(df)[0,1]]

    # Blend if subgroup model exists
    if input.HeadquartersState == "California":
        probs.append(models["california"].predict_proba(df)[0,1])
    if input.HeadquartersState == "Illinois":
        probs.append(models["illinois"].predict_proba(df)[0,1])
    if input.HeadquartersState == "Michigan":
        probs.append(models["michigan"].predict_proba(df)[0,1])
    if input.HeadquartersState == "New York":
        probs.append(models["newyork"].predict_proba(df)[0,1])
    if input.HeadquartersState == "Pennsylvania":
        probs.append(models["pennsylvania"].predict_proba(df)[0,1])
    if input.HeadquartersState == "Texas":
        probs.append(models["texas"].predict_proba(df)[0,1])
    if input.Sector == "Business Services":
        probs.append(models["business_services"].predict_proba(df)[0,1])
    if input.Sector == "Financials":
        probs.append(models["financials"].predict_proba(df)[0,1])
    if input.Sector == "Health Care":
        probs.append(models["health_care"].predict_proba(df)[0,1])
    if input.Sector == "Technology":
        probs.append(models["tech"].predict_proba(df)[0,1])
    if input.Industry == "Specialty Retailers: Other":
        probs.append(models["specialty_retailers"].predict_proba(df)[0,1])
    
    blended = sum(probs) / len(probs)
    print(f'probability scores: {probs}')

    df_log['predicted_prob'] = blended
    # Log user input
    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False, mode='w')
    else:
        df_log.to_csv(LOG_FILE, index=False, mode='a', header=False)
    # print(f"Predicted probability of stock price increase: {blended:.4f}")
    return {"probability": blended}
