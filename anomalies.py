import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import joblib
from features import load_and_preprocess_data, resample_to_hourly

def detect_isolation_forest_anomalies(df, target_col='Appliances', contamination=0.03):
    """
    Fits an Isolation Forest to detect general multivariate outliers.
    Returns the anomaly flags (-1 for anomalies, 1 for normal) and scores.
    """
    df_clean = df.copy()
    
    # Select key indicators for operational behavior
    # We want to identify abnormal target values relative to outdoor weather and internal temperature
    features = [target_col]
    if 'T_out' in df_clean.columns:
        features.append('T_out')
    if 'T1' in df_clean.columns:
        features.append('T1')
    if 'RH_1' in df_clean.columns:
        features.append('RH_1')
        
    X = df_clean[features].values
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_flags = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.decision_function(X)
    
    # Add to dataframe
    df_clean['is_anomaly_iso'] = (anomaly_flags == -1).astype(int)
    df_clean['anomaly_score_iso'] = anomaly_score_iso = anomaly_scores
    
    return df_clean, iso_forest

def detect_prediction_residual_anomalies(predictions_df, actual_col='Actual', pred_col='XGB_Pred', threshold_z=2.5):
    """
    Detects anomalies where the prediction error (residual) is statistically extreme.
    Good for flagging unexpected surges or drops compared to the forecast.
    """
    df_res = predictions_df.copy()
    
    # Calculate residuals
    df_res['residual'] = df_res[actual_col] - df_res[pred_col]
    
    # Calculate rolling mean and standard deviation of residuals to handle seasonal variance
    # Or use global statistics if preferred. Let's use a rolling window of 24 hours for local adaptivity.
    df_res['residual_mean'] = df_res['residual'].rolling(window=24, min_periods=1).mean()
    df_res['residual_std'] = df_res['residual'].rolling(window=24, min_periods=1).std().fillna(df_res['residual'].std())
    
    # Z-score of residuals
    df_res['residual_z'] = (df_res['residual'] - df_res['residual_mean']) / df_res['residual_std']
    df_res['residual_z'] = df_res['residual_z'].fillna(0)
    
    # Flag anomalies
    df_res['is_anomaly_residual'] = (np.abs(df_res['residual_z']) > threshold_z).astype(int)
    
    return df_res

def run_anomaly_detection_pipeline(data_path, save_dir='D:\\Smart_energy_predictor\\data'):
    """
    Runs and saves both Isolation Forest outliers and residual-based forecast surprises.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data for anomaly detection...")
    df = load_and_preprocess_data(data_path)
    df_hourly = resample_to_hourly(df)
    
    # 1. Run Isolation Forest
    print("Running Isolation Forest...")
    df_iso, model_iso = detect_isolation_forest_anomalies(df_hourly)
    
    # Save Isolation Forest Model
    joblib.dump(model_iso, os.path.join(save_dir, 'isolation_forest.joblib'))
    
    # 2. Run residual anomalies on model predictions if available
    predictions_file = os.path.join(save_dir, 'model_predictions.csv')
    if os.path.exists(predictions_file):
        print("Running prediction residual anomaly detection...")
        preds_df = pd.read_csv(predictions_file, index_col='date')
        
        # Decide which prediction model to use (XGBoost usually performs best)
        model_to_use = 'XGB_Pred' if 'XGB_Pred' in preds_df.columns else preds_df.columns[1]
        
        preds_anom_df = detect_prediction_residual_anomalies(preds_df, pred_col=model_to_use)
        preds_anom_df.to_csv(os.path.join(save_dir, 'anomaly_predictions_results.csv'))
        
        # Merge Isolation Forest flags back for full dataset representation
        # Slice df_iso to match predictions test dates
        test_dates = preds_anom_df.index
        df_iso_test = df_iso.loc[test_dates]
        
        full_anomaly_df = pd.concat([
            preds_anom_df,
            df_iso_test[['is_anomaly_iso', 'anomaly_score_iso']]
        ], axis=1)
        
        full_anomaly_df.to_csv(os.path.join(save_dir, 'anomaly_complete_results.csv'))
        print("Anomaly pipelines executed and saved successfully!")
        return full_anomaly_df
    else:
        # Save just Isolation Forest results if predictions are missing
        df_iso.to_csv(os.path.join(save_dir, 'anomaly_complete_results.csv'))
        print("Anomaly pipeline executed (Isolation Forest only) and saved successfully!")
        return df_iso

if __name__ == '__main__':
    data_path = "D:\\Smart_energy_predictor\\energydata_complete.csv"
    run_anomaly_detection_pipeline(data_path)
