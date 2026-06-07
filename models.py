import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from features import load_and_preprocess_data, resample_to_hourly, engineer_features, split_train_test_chronological

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error, handling zeros safely.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Mask out zeros to avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred):
    """
    Computes standard regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    return {
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'R2': round(r2, 3),
        'MAPE': round(mape, 3)
    }

def train_and_save_classical_models(data_path, save_dir='D:\\Smart_energy_predictor\\data'):
    """
    Loads data, engineers features, splits it, trains Linear Regression and XGBoost,
    evaluates them, and saves the models and performance metrics.
    """
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Pipeline: Load -> Resample -> Features -> Split
    print("Loading data in models pipeline...")
    df = load_and_preprocess_data(data_path)
    df_hourly = resample_to_hourly(df)
    df_feats = engineer_features(df_hourly)
    train_df, test_df = split_train_test_chronological(df_feats)
    
    # Define features and target
    # Exclude non-predictive metadata or the raw target
    target_col = 'Appliances'
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'target_lag_1', 'target_lag_2', 'target_lag_24',
        'target_roll_mean_3', 'target_roll_mean_24',
        'T_out', 'RH_out', 'Windspeed', 'Press_mm_hg', 'HDH', 'CDH',
        'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features.")
    
    # 2. Train Linear Regression
    print("Training Linear Regression baseline...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = evaluate_predictions(y_test, lr_preds)
    print("Linear Regression Metrics:", lr_metrics)
    
    # 3. Train XGBoost
    print("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = evaluate_predictions(y_test, xgb_preds)
    print("XGBoost Metrics:", xgb_metrics)
    
    # 4. Save models and metrics
    joblib.dump(lr_model, os.path.join(save_dir, 'linear_regression.joblib'))
    joblib.dump(xgb_model, os.path.join(save_dir, 'xgboost.joblib'))
    
    # Save test predictions for visualization in dashboard
    results_df = pd.DataFrame({
        'Actual': y_test,
        'LR_Pred': lr_preds,
        'XGB_Pred': xgb_preds
    }, index=test_df.index)
    
    results_df.to_csv(os.path.join(save_dir, 'model_predictions.csv'))
    
    # Save metadata about feature names & metrics
    metadata = {
        'features': feature_cols,
        'metrics': {
            'Linear Regression': lr_metrics,
            'XGBoost': xgb_metrics
        }
    }
    joblib.dump(metadata, os.path.join(save_dir, 'model_metadata.joblib'))
    
    print("Models and metadata saved successfully!")
    return metadata

if __name__ == '__main__':
    data_path = "D:\\Smart_energy_predictor\\energydata_complete.csv"
    train_and_save_classical_models(data_path)
