"""model_training.py

Train and evaluate an XGBoost regression model for baseline health risk.
"""
import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def prepare_xy(df, target_col, drop_cols=None):
    """Return X, y for modeling. Optionally drop identifier columns."""
    if drop_cols is None:
        drop_cols = ['State','District']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col].values
    return X, y

def train_xgb(df, target_col, model_path='models/xgb_model.pkl', test_size=0.2, random_state=42, **xgb_kwargs):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    X, y = prepare_xy(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = XGBRegressor(**xgb_kwargs) if xgb_kwargs else XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    # Save model
    joblib.dump(model, model_path)
    metrics = {'r2': float(r2), 'rmse': float(rmse)}
    return model, metrics

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
