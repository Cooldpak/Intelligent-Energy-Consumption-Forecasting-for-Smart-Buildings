import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path):
    """
    Loads the UCI appliances energy dataset and performs initial preprocessing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Convert date to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Drop random variables as they have no predictive power
    if 'rv1' in df.columns:
        df = df.drop(columns=['rv1', 'rv2'])
        
    return df

def resample_to_hourly(df):
    """
    Resamples 10-minute data to hourly frequency.
    Sums the energy consumption columns (Appliances, lights) to get total hourly Wh.
    Averages the temperature and humidity columns.
    """
    energy_cols = ['Appliances', 'lights']
    other_cols = [col for col in df.columns if col not in energy_cols]
    
    # Aggregate energy columns by sum (to get total Wh in the hour)
    df_energy = df[energy_cols].resample('h').sum()
    
    # Aggregate temperatures, humidities and weather variables by mean
    df_others = df[other_cols].resample('h').mean()
    
    # Combine back
    df_hourly = pd.concat([df_energy, df_others], axis=1)
    
    # Drop any rows with NaN caused by resampling gaps (if any)
    df_hourly = df_hourly.dropna()
    
    return df_hourly

def engineer_features(df, target_col='Appliances'):
    """
    Engineers time-series and domain-specific features.
    """
    df_feats = df.copy()
    
    # 1. Temporal features
    df_feats['hour'] = df_feats.index.hour
    df_feats['day_of_week'] = df_feats.index.dayofweek
    df_feats['month'] = df_feats.index.month
    df_feats['is_weekend'] = df_feats['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Lag features of the target
    # Predicting current hour t based on t-1, t-2, and t-24 values
    df_feats['target_lag_1'] = df_feats[target_col].shift(1)
    df_feats['target_lag_2'] = df_feats[target_col].shift(2)
    df_feats['target_lag_24'] = df_feats[target_col].shift(24)
    
    # 3. Rolling window features of the target
    df_feats['target_roll_mean_3'] = df_feats[target_col].shift(1).rolling(window=3).mean()
    df_feats['target_roll_mean_24'] = df_feats[target_col].shift(1).rolling(window=24).mean()
    
    # 4. Domain-specific HVAC features: Heating & Cooling Degree Hours
    # Using 18 degrees Celsius as standard comfort base temperature
    base_temp = 18.0
    if 'T_out' in df_feats.columns:
        df_feats['HDH'] = df_feats['T_out'].apply(lambda x: max(0, base_temp - x))
        df_feats['CDH'] = df_feats['T_out'].apply(lambda x: max(0, x - base_temp))
    else:
        # Fallback to T6 if T_out is missing
        df_feats['HDH'] = df_feats['T6'].apply(lambda x: max(0, base_temp - x))
        df_feats['CDH'] = df_feats['T6'].apply(lambda x: max(0, x - base_temp))
        
    # Drop rows with NaN values introduced by lags and rolling windows
    df_feats = df_feats.dropna()
    
    return df_feats

def split_train_test_chronological(df, train_ratio=0.8):
    """
    Splits the dataframe chronologically to prevent future data leakage in time-series forecasting.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

if __name__ == '__main__':
    # Test block
    data_path = "D:\\Smart_energy_predictor\\energydata_complete.csv"
    print("Loading data...")
    df = load_and_preprocess_data(data_path)
    print(f"Original shape: {df.shape}")
    
    print("Resampling to hourly...")
    df_hourly = resample_to_hourly(df)
    print(f"Hourly shape: {df_hourly.shape}")
    
    print("Engineering features...")
    df_feats = engineer_features(df_hourly)
    print(f"Features shape: {df_feats.shape}")
    
    train_df, test_df = split_train_test_chronological(df_feats)
    print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")
    print("Data loading and feature engineering works successfully!")
