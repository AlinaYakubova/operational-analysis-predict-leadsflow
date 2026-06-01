import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import config

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def main():
    print("=== RUNNING STATISTICAL MODELING (SARIMAX) ===")
    
    df = pd.read_csv("processed_monthly_global.csv", index_col=0, parse_dates=True)
    
    print("\n[EDA] Generating ACF and PACF plots for log-transformed series...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df['log_target'].dropna(), lags=24, ax=axes[0])
    plot_pacf(df['log_target'].dropna(), lags=24, ax=axes[1], method='ywm')
    plt.savefig("sarimax_acf_pacf.png", bbox_inches='tight')
    plt.close()
    print("ACF/PACF charts saved to 'sarimax_acf_pacf.png'")
    
    horizon = config.MONTHLY_TEST_HORIZON
    train_df = df.iloc[:-horizon]
    test_df = df.iloc[-horizon:]
    
    print(f"Train size: {len(train_df)} months, Test size: {len(test_df)} months.")
    
    # Using baseline parameters: Order (1,1,1) and Seasonal Order (1,1,1,12)
    # Including log_exog (marketing spend) as exogenous variable
    print("\n[Modeling] Fitting SARIMAX model with exogenous marketing spend...")
    
    model = SARIMAX(
        train_df['log_target'],
        exog=train_df['log_exog'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False)
    print(results.summary())
    
    # Out-of-sample Forecast
    print(f"\n[Forecasting] Predicting next {horizon} months...")
    forecast_log = results.forecast(steps=horizon, exog=test_df['log_exog'])
    
    # Inverse transformation from log space back to original scale (Fil Rouge rule)
    forecast_final = np.exp(forecast_log) - 1
    actual_final = test_df[config.TARGET_COL].values
    
    # Save predictions for the final evaluation and Diebold-Mariano test
    predictions_df = pd.DataFrame({
        'Actual': actual_final,
        'SARIMAX_Forecast': forecast_final
    }, index=test_df.index)
    predictions_df.to_csv("predictions_sarimax.csv")
    
    # Evaluate Performance
    mae, rmse, mape = calculate_metrics(actual_final, forecast_final)
    print("\n=== SARIMAX TEST PERFORMANCE ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")
    
    # Plot Forecast vs Actuals
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[config.TARGET_COL], label="Historical Actuals", color="blue")
    plt.plot(test_df.index, forecast_final, label="SARIMAX Forecast", color="red", linestyle="--")
    plt.title("Kodland Lead Volume Forecasting - SARIMAX")
    plt.xlabel("Timeline")
    plt.ylabel("Leads Count")
    plt.legend()
    plt.savefig("sarimax_forecast_vs_actuals.png", bbox_inches='tight')
    plt.close()
    print("Forecast visual chart saved to 'sarimax_forecast_vs_actuals.png'")

if __name__ == "__main__":
    main()