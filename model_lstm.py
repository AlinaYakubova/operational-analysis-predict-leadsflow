import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import config

torch.manual_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)


def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])


def fit_scalers(train_df):
    scaler_target = MinMaxScaler()
    scaler_exog = MinMaxScaler()
    scaler_target.fit(train_df[['log_target']])
    scaler_exog.fit(train_df[['log_exog']])
    return scaler_target, scaler_exog


def build_sequences(scaled_target, scaled_exog, fourier_sin, fourier_cos, n_lags):
    sequences, targets = [], []
    for i in range(len(scaled_target) - n_lags):
        seq = np.column_stack((
            scaled_target[i:i + n_lags],
            scaled_exog[i:i + n_lags],
            fourier_sin[i:i + n_lags],
            fourier_cos[i:i + n_lags]
        ))
        sequences.append(seq)
        targets.append(scaled_target[i + n_lags])
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)
    return X, y


def train_model(X_train, y_train, hidden_size=32, num_layers=1, epochs=200, lr=0.001):
    model = LSTMRegressor(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], MSE Loss: {loss.item():.6f}")

    return model


def sliding_window_forecast(model, train_scaled_target, train_scaled_exog,
                             train_fourier_sin, train_fourier_cos,
                             test_scaled_exog, test_fourier_sin, test_fourier_cos,
                             n_lags, horizon):
    model.eval()
    forecast_scaled = []
    current_target_window = list(train_scaled_target[-n_lags:])
    current_exog_window = list(train_scaled_exog[-n_lags:])
    current_sin_window = list(train_fourier_sin[-n_lags:])
    current_cos_window = list(train_fourier_cos[-n_lags:])

    for i in range(horizon):
        input_seq = np.column_stack((
            current_target_window,
            current_exog_window,
            current_sin_window,
            current_cos_window
        ))
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            next_pred_scaled = model(input_tensor).item()

        forecast_scaled.append(next_pred_scaled)
        current_target_window.pop(0)
        current_target_window.append(next_pred_scaled)
        current_exog_window.pop(0)
        current_exog_window.append(test_scaled_exog[i])
        current_sin_window.pop(0)
        current_sin_window.append(test_fourier_sin[i])
        current_cos_window.pop(0)
        current_cos_window.append(test_fourier_cos[i])

    return np.array(forecast_scaled)


def inverse_forecast(forecast_scaled, scaler_target):
    forecast_log = scaler_target.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    return np.exp(forecast_log) - 1


def plot_forecast(df, test_dates, actual_final, forecast_final, horizon,
                  save_path="lstm_forecast_vs_actuals.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[:-horizon], df[config.TARGET_COL].iloc[:-horizon],
             label="Train Actuals", color="blue")
    plt.plot(test_dates, actual_final, label="Test Actuals", color="cyan")
    plt.plot(test_dates, forecast_final, label="LSTM Sliding Forecast", color="purple", linestyle="--")
    plt.title("Lead Volume Forecasting - LSTM Neural Network")
    plt.xlabel("Timeline")
    plt.ylabel("Leads Count")
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Forecast visual chart saved to '{save_path}'")


def main():
    print("=== RUNNING NEURAL NETWORK MODULE (LSTM ON PYTORCH) ===")

    df = pd.read_csv("processed_weekly_global.csv", index_col=0, parse_dates=True)

    n_lags = 12
    horizon = config.WEEKLY_TEST_HORIZON
    train_df = df.iloc[:-horizon]
    test_df = df.iloc[-horizon:]

    scaler_target, scaler_exog = fit_scalers(train_df)
    train_scaled_target = scaler_target.transform(train_df[['log_target']]).flatten()
    train_scaled_exog = scaler_exog.transform(train_df[['log_exog']]).flatten()
    test_scaled_exog = scaler_exog.transform(test_df[['log_exog']]).flatten()

    train_fourier_sin = train_df['fourier_sin_1'].values
    train_fourier_cos = train_df['fourier_cos_1'].values
    test_fourier_sin = test_df['fourier_sin_1'].values
    test_fourier_cos = test_df['fourier_cos_1'].values

    X_train, y_train = build_sequences(
        train_scaled_target, train_scaled_exog,
        train_fourier_sin, train_fourier_cos,
        n_lags
    )
    print(f"LSTM Training Tensor Shape: {X_train.shape}")

    print("\n[Training] Training LSTM model over epochs...")
    model = train_model(X_train, y_train)

    print(f"\n[Forecasting] Generating iterative sliding forecast for {horizon} weeks...")
    forecast_scaled = sliding_window_forecast(
        model,
        train_scaled_target, train_scaled_exog,
        train_fourier_sin, train_fourier_cos,
        test_scaled_exog, test_fourier_sin, test_fourier_cos,
        n_lags, horizon
    )
    forecast_final = inverse_forecast(forecast_scaled, scaler_target)

    test_dates = df.index[-horizon:]
    actual_final = df.loc[test_dates, config.TARGET_COL].values

    pd.DataFrame({
        'Actual': actual_final,
        'LSTM_Forecast': forecast_final
    }, index=test_dates).to_csv("predictions_lstm.csv")

    mae, rmse, mape = calculate_metrics(actual_final, forecast_final)
    print("\n=== LSTM TEST PERFORMANCE ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    plot_forecast(df, test_dates, actual_final, forecast_final, horizon)


if __name__ == "__main__":
    main()
