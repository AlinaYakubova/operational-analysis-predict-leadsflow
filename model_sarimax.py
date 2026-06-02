import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import config


def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def plot_acf_pacf(series, save_path="sarimax_acf_pacf.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=24, ax=axes[0])
    plot_pacf(series.dropna(), lags=24, ax=axes[1], method='ywm')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ACF/PACF charts saved to '{save_path}'")

def find_best_order(train_df, seasonal_d):
    auto_model = pm.auto_arima(
        train_df['log_target'],
        exogenous=train_df[['log_exog']],
        seasonal=True, m=12,
        d=None, D=seasonal_d,
        start_p=0, max_p=2,
        start_q=0, max_q=2,
        start_P=0, max_P=1,
        start_Q=0, max_Q=1,
        information_criterion='aic',
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    print(f"Best order found: {auto_model.order}, seasonal_order: {auto_model.seasonal_order}")
    print(f"auto_arima AIC: {auto_model.aic():.2f}")
    return auto_model.order, auto_model.seasonal_order

def fit_sarimax(train_df, order, seasonal_order):
    model = SARIMAX(
        train_df['log_target'],
        exog=train_df['log_exog'],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print(results.summary())
    print(f"\nAIC: {results.aic:.2f}")
    return results


def save_diagnostics(results, save_path="sarimax_diagnostics.png"):
    results.plot_diagnostics(figsize=(10, 6))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Model diagnostics saved to '{save_path}'")

def generate_forecast(results, horizon, test_df):
    forecast_obj = results.get_forecast(steps=horizon, exog=test_df['log_exog'])
    forecast_final = np.exp(forecast_obj.predicted_mean) - 1
    return forecast_final

def plot_forecast_comparison(df, test_df, forecasts, titles, save_path="sarimax_forecast_vs_actuals.png"):
    fig, axes = plt.subplots(len(forecasts), 1, figsize=(10, 5 * len(forecasts)))

    for ax, forecast, title in zip(axes, forecasts, titles):
        ax.plot(df.index, df[config.TARGET_COL], label="Historical Actuals", color="blue")
        ax.plot(test_df.index, forecast, label="SARIMAX Forecast", color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Timeline")
        ax.set_ylabel("Leads Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Forecast comparison chart saved to '{save_path}'")

def main():
    print("=== RUNNING STATISTICAL MODELING (SARIMAX) ===")

    df = pd.read_csv("processed_monthly_global.csv", index_col=0, parse_dates=True)

    print("\n[EDA] Generating ACF and PACF plots for log-transformed series...")
    plot_acf_pacf(df['log_target'])

    horizon = config.MONTHLY_TEST_HORIZON
    train_df = df.iloc[:-horizon]
    test_df = df.iloc[-horizon:]
    print(f"Train size: {len(train_df)} months, Test size: {len(test_df)} months.")

    actual_final = test_df[config.TARGET_COL].values

    print("\n[Modeling] Fitting manual SARIMAX (1,1,1)(1,1,1,12)...")
    results_manual = fit_sarimax(train_df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    save_diagnostics(results_manual)
    forecast_manual = generate_forecast(results_manual, horizon, test_df)

    mae, rmse, mape = calculate_metrics(actual_final, forecast_manual)
    print("\n=== MANUAL SARIMAX (1,1,1)(1,1,1,12) PERFORMANCE ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    forecasts = [forecast_manual]
    titles = ["SARIMAX (1,1,1)(1,1,1,12) — Manual from ACF/PACF"]
    predictions = {'Actual': actual_final, 'SARIMAX_Manual': forecast_manual}

    for d_val in [0, 1]:
        print(f"\n[Parameter Selection] Running auto_arima with D={d_val}...")
        order_auto, seasonal_order_auto = find_best_order(train_df, seasonal_d=d_val)

        print(f"\n[Modeling] Fitting auto_arima SARIMAX with D={d_val}...")
        results_auto = fit_sarimax(train_df, order_auto, seasonal_order_auto)
        forecast_auto = generate_forecast(results_auto, horizon, test_df)

        mae, rmse, mape = calculate_metrics(actual_final, forecast_auto)
        print(f"\n=== AUTO_ARIMA D={d_val}: {order_auto}{seasonal_order_auto} PERFORMANCE ===")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : {mape:.2f}%")

        forecasts.append(forecast_auto)
        titles.append(f"SARIMAX {order_auto}{seasonal_order_auto} — auto_arima D={d_val}")
        predictions[f'SARIMAX_Auto_D{d_val}'] = forecast_auto

    pd.DataFrame(predictions, index=test_df.index).to_csv("predictions_sarimax.csv")

    print(f"\n[Forecasting] Plotting comparison for {horizon} months...")
    plot_forecast_comparison(df, test_df, forecasts, titles)


if __name__ == "__main__":
    main()