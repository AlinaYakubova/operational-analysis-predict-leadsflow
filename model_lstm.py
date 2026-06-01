import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import config

# Set seed for PyTorch reproducibility
torch.manual_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

# Define LSTM Network Structure strictly matching PyTorch guidelines
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Take the output of the very last time step
        last_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_step_out)
        return predictions

def main():
    print("=== RUNNING NEURAL NETWORK MODULE (LSTM ON PYTORCH) ===")
    
    df = pd.read_csv("processed_weekly_global.csv", index_col=0, parse_dates=True)
    
    # Scale features via MinMaxScaler
    scaler_target = MinMaxScaler()
    scaler_exog = MinMaxScaler()
    
    df['scaled_target'] = scaler_target.fit_transform(df[['log_target']])
    df['scaled_exog'] = scaler_exog.fit_transform(df[['log_exog']])
    
    # Train-Test Split based on weekly horizon
    horizon = config.WEEKLY_TEST_HORIZON
    train_df = df.iloc[:-horizon]
    test_df = df.iloc[-horizon:]
    
    # Prepare sliding sequences for LSTM training
    n_lags = 4
    train_sequences = []
    train_targets = []
    
    scaled_target_values = train_df['scaled_target'].values
    scaled_exog_values = train_df['scaled_exog'].values
    
    for i in range(len(train_df) - n_lags):
        # Window contains: [target_t-4, exog_t-4], ..., [target_t-1, exog_t-1]
        seq_features = np.column_stack((scaled_target_values[i:i+n_lags], scaled_exog_values[i:i+n_lags]))
        # Target to predict: target_t
        target_val = scaled_target_values[i+n_lags]
        
        train_sequences.append(seq_features)
        train_targets.append(target_val)
        
    X_train = torch.tensor(np.array(train_sequences), dtype=torch.float32)
    y_train = torch.tensor(np.array(train_targets), dtype=torch.float32).unsqueeze(1)
    
    print(f"LSTM Training Tensor Shape: {X_train.shape}")
    
    # Initialize Network, Loss function and Optimizer
    model = LSTMRegressor(input_size=2, hidden_size=16, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    print("\n[Training] Training LSTM model over epochs...")
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch [{epoch+1}/150], MSE Loss: {loss.item():.6f}")
            
    # Sliding Multi-Step Forecast
    print(f"\n[Forecasting] Generating iterative sliding forecast for {horizon} weeks...")
    model.eval()
    
    forecast_scaled = []
    # Initialize rolling window with the last 'n_lags' points from training set
    current_target_window = list(scaled_target_values[-n_lags:])
    current_exog_window = list(scaled_exog_values[-n_lags:])
    
    # Exogenous values for the future test period
    all_scaled_exog = df['scaled_exog'].values
    test_exog_future = all_scaled_exog[-horizon:]
    
    for i in range(horizon):
        # Combine target and exog into a single sequence matrix
        input_seq = np.column_stack((current_target_window, current_exog_window))
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0) # add batch dim
        
        with torch.no_grad():
            next_pred_scaled = model(input_tensor).item()
            
        forecast_scaled.append(next_pred_scaled)
        
        # Update sliding target window: drop oldest, push new prediction
        current_target_window.pop(0)
        current_target_window.append(next_pred_scaled)
        
        # Update sliding exogenous window using actual test spend values
        current_exog_window.pop(0)
        current_exog_window.append(test_exog_future[i])
        
    # Inverse transformations back to original scale
    forecast_scaled_arr = np.array(forecast_scaled).reshape(-1, 1)
    forecast_log = scaler_target.inverse_transform(forecast_scaled_arr).flatten()
    forecast_final = np.exp(forecast_log) - 1
    
    test_dates = df.index[-horizon:]
    actual_final = df.loc[test_dates, config.TARGET_COL].values
    
    # Save predictions for statistical validation module
    predictions_df = pd.DataFrame({
        'Actual': actual_final,
        'LSTM_Forecast': forecast_final
    }, index=test_dates)
    predictions_df.to_csv("predictions_lstm.csv")
    
    # Evaluate
    mae, rmse, mape = calculate_metrics(actual_final, forecast_final)
    print("\n=== LSTM TEST PERFORMANCE ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")
    
    # Visualization chart
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[:-horizon], df[config.TARGET_COL].iloc[:-horizon], label="Train Actuals", color="blue")
    plt.plot(test_dates, actual_final, label="Test Actuals", color="cyan")
    plt.plot(test_dates, forecast_final, label="LSTM Sliding Forecast", color="purple", linestyle="--")
    plt.title("Kodland Lead Volume Forecasting - LSTM Neural Network")
    plt.xlabel("Timeline")
    plt.ylabel("Leads Count")
    plt.legend()
    plt.savefig("lstm_forecast_vs_actuals.png", bbox_inches='tight')
    plt.close()
    print("Forecast visual chart saved to 'lstm_forecast_vs_actuals.png'")

if __name__ == "__main__":
    main()