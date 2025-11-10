"""data_preprocessing.py

Functions to load, clean, merge and scale datasets used in the PHRS pipeline.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_csv(path):
    """Load a CSV file into a DataFrame. Raises if file not found."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def merge_datasets(aqi_df, weather_df, smoking_df, beds_df=None, on_cols=('State','District')):
    """Merge multiple dataframes on the specified keys. Returns merged DataFrame."""
    df = aqi_df.merge(weather_df, on=list(on_cols), how='outer')
    df = df.merge(smoking_df, on=list(on_cols), how='left')
    if beds_df is not None:
        df = df.merge(beds_df, on=list(on_cols), how='left')
    return df

def clean_dataframe(df, required_cols=None):
    """Basic cleaning: strip column names, handle missing columns, drop rows missing required_cols."""
    # Normalize column names
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    if required_cols is None:
        required_cols = ['PM2.5','PM10','AQI','Temperature','Humidity','Wind_Speed','Rainfall','Male_Smoking_%','Female_Smoking_%']
    # Drop rows missing essential measurements
    present = [c for c in required_cols if c in df.columns]
    if present:
        df = df.dropna(subset=present)
    # Fill other numeric missing values with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    return df

def scale_features(df, feature_cols):
    """Scale selected numeric columns to [0,1] using MinMaxScaler."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    return df_scaled, scaler

def save_processed(df, out_path):
    """Save processed dataframe to CSV (creates folder if required)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
