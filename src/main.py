"""main.py

Pipeline runner for PHRS model. Update the file paths below to match your data locations.
"""
import os
from src.data_preprocessing import load_csv, merge_datasets, clean_dataframe, scale_features, save_processed
from src.model_training import train_xgb, prepare_xy, load_model
from src.risk_calculation import compute_environmental_risk, compute_phrs
from src.visualization import plot_top_risk, plot_low_risk, plot_statewise_box

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def run_pipeline():
    # --- Edit these paths if your filenames are different ---
    aqi_path = os.path.join(DATA_DIR, 'aqi.csv')
    weather_path = os.path.join(DATA_DIR, 'weather.csv')
    smoking_path = os.path.join(DATA_DIR, 'smoking.csv')
    beds_path = os.path.join(DATA_DIR, 'beds_per_1000.csv')  # optional

    # Load inputs (these files are *examples* — replace with your real file names)
    try:
        aqi = load_csv(aqi_path)
        weather = load_csv(weather_path)
        smoking = load_csv(smoking_path)
    except FileNotFoundError as e:
        print('Data files not found. Please place your CSVs in data/raw/ and update filenames in main.py.')
        print(str(e))
        return

    # Merge, clean, and compute
    df = merge_datasets(aqi, weather, smoking, beds_df=None)
    df = clean_dataframe(df)
    # Compute environmental risk and PHRS
    df = compute_environmental_risk(df)
    df = compute_phrs(df)
    # Save processed output
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_csv = os.path.join(PROCESSED_DIR, 'final_phrs.csv')
    df.to_csv(out_csv, index=False)
    print(f'Processed PHRS saved to: {out_csv}')

    # Optional: train model if target column exists
    if 'Baseline_Health_Risk' in df.columns:
        model_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'xgb_model.pkl')
        model, metrics = train_xgb(df, target_col='Baseline_Health_Risk', model_path=model_path)
        print('Trained XGBoost model metrics:', metrics)
    else:
        print('Baseline_Health_Risk target not present — skipping model training.')

    # Visualizations (only if District and State exist)
    if 'District' in df.columns:
        plot_top_risk(df)
        plot_low_risk(df)
    if 'State' in df.columns:
        plot_statewise_box(df)
    print('Visualizations created under results/figures/ (if data available).')

if __name__ == '__main__':
    run_pipeline()
