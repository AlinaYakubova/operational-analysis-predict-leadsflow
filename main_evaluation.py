import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def select_best_prediction(df, actual_col='Actual'):
    prediction_cols = [col for col in df.columns if col != actual_col]
    if not prediction_cols:
        raise ValueError(f"No prediction columns found in dataframe. Columns: {list(df.columns)}")

    actual = df[actual_col].values
    scores = {
        col: calculate_metrics(actual, df[col].values)
        for col in prediction_cols
    }
    best_col = min(scores, key=lambda col: scores[col][2])
    return best_col, df[best_col].values, scores[best_col]

def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Performs the Diebold-Mariano test for predictive accuracy.
    H0: Both models have the same forecast accuracy.
    H1: The two models have statistically different accuracy levels.
    """
    T = len(actual)
    # Calculate loss differentials (using absolute error loss)
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d = e1 - e2
    
    # Mean of loss differential
    d_mean = np.mean(d)
    
    # Calculate autocovariances for the variance of d (Newey-West type estimator for lag h-1)
    gamma = np.zeros(h)
    for lag in range(0, h):
        gamma[lag] = np.mean((d[:T-lag] - d_mean) * (d[lag:] - d_mean))
        
    # Variance estimation
    var_d = gamma[0] + 2 * np.sum([((h - j) / h) * gamma[j] for j in range(1, h)]) if h > 1 else gamma[0]
    
    # Diebold-Mariano Statistic
    dm_stat = d_mean / np.sqrt(var_d / T)
    
    # Two-sided p-value from standard normal distribution
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def main():
    print("=== RUNNING FINAL EVALUATION & STATISTICAL TESTING MODULE ===")
    
    # 1. Load predictions from saved files
    try:
        sarimax_df = pd.read_csv("predictions_sarimax.csv", index_col=0, parse_dates=True)
        xgboost_df = pd.read_csv("predictions_xgboost.csv", index_col=0, parse_dates=True)
        lstm_df = pd.read_csv("predictions_lstm.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("[Error] Please make sure you have run model_sarimax.py, model_xgboost.py, and model_lstm.py first!")
        return

    # 2. Print consolidated metrics summary
    print("\n========================================================")
    print("               CONSOLIDATED METRICS REPORT              ")
    print("========================================================")
    
    # SARIMAX Metrics (Monthly Scale)
    s_act = sarimax_df['Actual'].values
    sarimax_col, s_pred, (s_mae, s_rmse, s_mape) = select_best_prediction(sarimax_df)
    print(f"Model 1: SARIMAX ({sarimax_col} | Monthly Frequency | H = 12)")
    print(f"  MAE:  {s_mae:.2f}")
    print(f"  RMSE: {s_rmse:.2f}")
    print(f"  MAPE: {s_mape:.2f}%")
    
    # XGBoost Metrics (Weekly Scale)
    x_act = xgboost_df['Actual'].values
    xgboost_col, x_pred, (x_mae, x_rmse, x_mape) = select_best_prediction(xgboost_df)
    print(f"\nModel 2: XGBoost + Optuna ({xgboost_col} | Weekly Frequency | H = 52)")
    print(f"  MAE:  {x_mae:.2f}")
    print(f"  RMSE: {x_rmse:.2f}")
    print(f"  MAPE: {x_mape:.2f}%")
    
    # LSTM Metrics (Weekly Scale)
    l_act, l_pred = lstm_df['Actual'].values, lstm_df['LSTM_Forecast'].values
    l_mae, l_rmse, l_mape = calculate_metrics(l_act, l_pred)
    print(f"\nModel 3: LSTM Neural Network (Weekly Frequency | H = 52)")
    print(f"  MAE:  {l_mae:.2f}")
    print(f"  RMSE: {l_rmse:.2f}")
    print(f"  MAPE: {l_mape:.2f}%")
    print("========================================================")

    # 3. Perform Diebold-Mariano Test
    print("\n[Hypothesis Testing] Conducting Diebold-Mariano Test (XGBoost vs LSTM)...")
    
    # h=1 as we evaluate point forecasts sequence iteratively
    dm_stat, p_val = diebold_mariano_test(x_act, x_pred, l_pred, h=1)
    
    print(f"  Diebold-Mariano Statistic: {dm_stat:.4f}")
    print(f"  p-value: {p_val:.4f}")
    
    if p_val <= 0.05:
        print("=> RESULT: Reject H0. The difference in forecast accuracy is STATISTICALLY SIGNIFICANT.")
        if dm_stat < 0:
            print("   XGBoost statistically outperforms LSTM with systematic confidence.")
        else:
            print("   LSTM statistically outperforms XGBoost with systematic confidence.")
    else:
        print("=> RESULT: Fail to reject H0. The difference in accuracy between XGBoost and LSTM is purely random.")

    # 4. Generate Final Comparative Plot for Weekly Models
    plot_xgboost_df = xgboost_df.iloc[:-1]
    plot_lstm_df = lstm_df.iloc[:-1]
    plot_x_pred = x_pred[:-1]
    plot_l_pred = l_pred[:-1]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_xgboost_df.index, plot_xgboost_df['Actual'], label="Actual Leads (CRM)", color="black", linewidth=2)
    plt.plot(plot_xgboost_df.index, plot_x_pred, label=f"XGBoost {xgboost_col} (MAPE: {x_mape:.2f}%)", color="green", linestyle="--")
    plt.plot(plot_lstm_df.index, plot_l_pred, label=f"LSTM (MAPE: {l_mape:.2f}%)", color="purple", linestyle=":")
    
    plt.title("Kodland Lead Volume Forecasting - Final Model Comparison (Weekly Test Set)")
    plt.xlabel("Timeline (Weeks)")
    plt.ylabel("Leads Count")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("final_model_comparison.png", bbox_inches='tight')
    plt.close()
    print("\n[Visualization] Final comparative chart saved to 'final_model_comparison.png'")

if __name__ == "__main__":
    main()
