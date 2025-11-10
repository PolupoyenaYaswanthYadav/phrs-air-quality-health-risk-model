"""risk_calculation.py

Compute environmental risk components and the final PHRS scores for different demographic groups.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def compute_environmental_risk(df, beta_aqi=0.004, beta_temp=0.02):
    """Compute an environmental stressor risk based on AQI and temperature and mitigation from wind/rain."""
    df = df.copy()
    # Ensure columns exist, use fallbacks if necessary
    aqi_col = 'AQI' if 'AQI' in df.columns else 'PM2.5'
    temp_col = 'Temperature' if 'Temperature' in df.columns else 'Temp'
    wind_col = 'Wind_Speed' if 'Wind_Speed' in df.columns else 'Wind'
    rain_col = 'Rainfall' if 'Rainfall' in df.columns else 'Rain'

    aqi = df[aqi_col].astype(float)
    temp = df[temp_col].astype(float)
    wind = df[wind_col].astype(float)
    rain = df[rain_col].astype(float)

    aqi_excess = np.exp(beta_aqi * np.maximum(0, aqi - 50)) - 1
    temp_excess = np.exp(beta_temp * np.maximum(0, np.abs(temp - 20))) - 1
    total_stressor = aqi_excess + temp_excess

    # Prevent division by zero
    wind_mean = np.maximum(wind.mean(), 1e-6)
    rain_mean = np.maximum(rain.mean(), 1e-6)

    wind_mitigation = np.exp(-0.5 * wind / wind_mean)
    rain_mitigation = np.exp(-0.8 * rain / rain_mean)

    df['Env_Risk_Raw'] = total_stressor * wind_mitigation * rain_mitigation
    # Normalize Env_Risk_Raw to 0-1 for later steps
    scaler = MinMaxScaler()
    df['Env_Risk'] = scaler.fit_transform(df[['Env_Risk_Raw']])
    return df

def compute_phrs(df, female_factor=1.17, s_weight=0.5):
    """Compute PHRS for Male/Female and Smoker/Non-Smoker groups and scale to 1-100."""
    df = df.copy()
    # Safeguard column names
    df['Male_Smoking_%'] = df.get('Male_Smoking_%', df.get('Male_Smoking_rate', 0)).astype(float)
    df['Female_Smoking_%'] = df.get('Female_Smoking_%', df.get('Female_Smoking_rate', 0)).astype(float)
    df['Beds_per_1000'] = df.get('Beds_per_1000', df.get('Beds_per_1000', 0)).astype(float)

    # Behavior penalties (log1p dampens extreme values)
    df['Male_Smoke_Penalty'] = np.log1p(df['Male_Smoking_%'] * s_weight / 100.0)
    df['Female_Smoke_Penalty'] = np.log1p(df['Female_Smoking_%'] * s_weight / 100.0)

    # Regional modifier: higher beds -> lower risk (use percentile rank)
    df['Reg_Mod'] = df['Beds_per_1000'].rank(pct=True).fillna(0.0)

    # Base non-smoker scores: Env_Risk * (1 - Reg_Mod + Behav_mod) ; Behav_mod for non-smoker = 0
    df['PHRS_Male_NonSmoker_raw'] = df['Env_Risk'] * (1.0 - df['Reg_Mod'])
    df['PHRS_Female_NonSmoker_raw'] = df['PHRS_Male_NonSmoker_raw'] * female_factor

    # Smoker scores add the smoking penalty
    df['PHRS_Male_Smoker_raw'] = df['Env_Risk'] * (1.0 + df['Male_Smoke_Penalty'] - df['Reg_Mod'])
    df['PHRS_Female_Smoker_raw'] = df['Env_Risk'] * (1.0 + df['Female_Smoke_Penalty'] - df['Reg_Mod']) * female_factor

    # Scale raw PHRS to 1-100 globally across the dataset
    phrs_cols = ['PHRS_Male_Smoker_raw','PHRS_Male_NonSmoker_raw','PHRS_Female_Smoker_raw','PHRS_Female_NonSmoker_raw']
    mins = df[phrs_cols].min().min()
    maxs = df[phrs_cols].max().max()
    range_val = max(maxs - mins, 1e-6)

    for col in phrs_cols:
        scaled = 1.0 + 99.0 * (df[col] - mins) / range_val
        df[col.replace('_raw','')] = scaled

    df['Overall_Avg_Risk'] = df[['PHRS_Male_Smoker','PHRS_Male_NonSmoker','PHRS_Female_Smoker','PHRS_Female_NonSmoker']].mean(axis=1)
    return df
