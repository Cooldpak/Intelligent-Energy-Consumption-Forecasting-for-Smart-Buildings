import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from features import load_and_preprocess_data, resample_to_hourly, engineer_features, split_train_test_chronological
from models import evaluate_predictions

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class EnergyDataset(Dataset):
    """
    Custom PyTorch Dataset for creating sliding window sequences.
    Given a feature matrix and target vector, it returns sequences of length seq_length
    for inputs, and the next-step value for the target.
    """
    def __init__(self, X, y, seq_length=24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
        
    def __getitem__(self, idx):
        # Slice a sequence of shape (seq_length, num_features)
        X_seq = self.X[idx : idx + self.seq_length]
        # Target is the value immediately following the sequence
        y_target = self.y[idx + self.seq_length]
        return X_seq, y_target

class LSTMModel(nn.Module):
    """
    PyTorch Recurrent Neural Network using LSTM cells for time-series regression.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch_size, seq_len, num_features)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected regression head
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden and cell states to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Extract the hidden state of the very last time step in the sequence
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1) # shape: (batch_size,)

def train_lstm_model(data_path, save_dir='D:\\Smart_energy_predictor\\data', epochs=10, batch_size=32, seq_length=24):
    """
    Pipeline to scale data, create sequences, train PyTorch LSTM, evaluate, and save models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Pipeline: Load -> Resample -> Features -> Split
    print("Loading data in LSTM pipeline...")
    df = load_and_preprocess_data(data_path)
    df_hourly = resample_to_hourly(df)
    df_feats = engineer_features(df_hourly)
    train_df, test_df = split_train_test_chronological(df_feats)
    
    target_col = 'Appliances'
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'target_lag_1', 'target_lag_2', 'target_lag_24',
        'target_roll_mean_3', 'target_roll_mean_24',
        'T_out', 'RH_out', 'Windspeed', 'Press_mm_hg', 'HDH', 'CDH',
        'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3'
    ]
    
    # Extract raw arrays
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[target_col].values
    
    # 2. Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # 3. Create PyTorch datasets and loaders
    train_dataset = EnergyDataset(X_train_scaled, y_train_scaled, seq_length)
    test_dataset = EnergyDataset(X_test_scaled, y_test_scaled, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LSTM on device: {device}")
    
    model = LSTMModel(input_dim=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.6f}")
        
    # 6. Evaluation
    model.eval()
    lstm_scaled_preds = []
    actual_scaled_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            lstm_scaled_preds.extend(outputs.cpu().numpy())
            actual_scaled_targets.extend(y_batch.numpy())
            
    # Convert predictions back to original scale
    lstm_preds = scaler_y.inverse_transform(np.array(lstm_scaled_preds).reshape(-1, 1)).flatten()
    
    # The actual values corresponding to the sequences (excluding the initial seq_length points)
    # The sequences are length seq_length, so we drop the first seq_length records of the test target
    actual_targets = y_test_raw[seq_length:]
    
    lstm_metrics = evaluate_predictions(actual_targets, lstm_preds)
    print("LSTM Metrics:", lstm_metrics)
    
    # 7. Save model, scalers, predictions, and metrics
    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_weights.pth'))
    joblib.dump(scaler_X, os.path.join(save_dir, 'lstm_scaler_X.joblib'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'lstm_scaler_y.joblib'))
    
    # Append LSTM predictions to the model_predictions.csv if it exists
    predictions_file = os.path.join(save_dir, 'model_predictions.csv')
    if os.path.exists(predictions_file):
        existing_preds = pd.read_csv(predictions_file, index_col='date')
        # We align by slicing the indices, since LSTM needs the first 24 points as history,
        # its predictions start 24 hours later than the test set start
        lstm_series = pd.Series(lstm_preds, index=existing_preds.index[seq_length:], name='LSTM_Pred')
        updated_preds = pd.concat([existing_preds, lstm_series], axis=1)
        updated_preds.to_csv(predictions_file)
    else:
        # Fallback
        results_df = pd.DataFrame({
            'Actual': actual_targets,
            'LSTM_Pred': lstm_preds
        }, index=test_df.index[seq_length:])
        results_df.to_csv(predictions_file)
        
    # Update metadata
    metadata_file = os.path.join(save_dir, 'model_metadata.joblib')
    if os.path.exists(metadata_file):
        metadata = joblib.load(metadata_file)
    else:
        metadata = {'features': feature_cols, 'metrics': {}}
        
    metadata['metrics']['LSTM (PyTorch)'] = lstm_metrics
    joblib.dump(metadata, metadata_file)
    
    print("LSTM Model weights and scaling configs saved successfully!")
    return lstm_metrics

if __name__ == '__main__':
    data_path = "D:\\Smart_energy_predictor\\energydata_complete.csv"
    train_lstm_model(data_path, epochs=5) # 5 epochs for testing
