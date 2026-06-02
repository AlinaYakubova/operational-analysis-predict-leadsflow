import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import config

def load_and_aggregate_data(file_path, time_col):
    df = pd.read_csv(file_path)
    df[time_col] = pd.to_datetime(df[time_col])
    agg_df = df.groupby(time_col)[[config.TARGET_COL, config.EXOG_COL]].sum().sort_index()
    return agg_df

def check_stationarity(series, name="Series"):
    print(f"\n--- Augmented Dickey-Fuller Test for: {name} ---")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    
    if result[1] <= 0.05:
        print("=> The series is STATIONARY (Reject H0)")
        return True
    else:
        print("=> The series is NON-STATIONARY (Fail to reject H0)")
        return False

def apply_log_transform(df):
    df['log_target'] = np.log1p(df[config.TARGET_COL])
    df['log_exog'] = np.log1p(df[config.EXOG_COL])
    return df

def detect_and_handle_outliers(df, log_col, period, name="Series"):

    print(f"\n--- Outlier Detection via Decomposition Residuals for: {name} ---")

    decomp = seasonal_decompose(df[log_col], model='additive', period=period)
    residuals = decomp.resid.dropna()

    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr

    outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
    outlier_dates = residuals[outlier_mask].index
    print(f"Outliers detected: {outlier_mask.sum()} points out of {len(residuals)}")

    df['outlier_flag'] = 0
    df.loc[outlier_dates, 'outlier_flag'] = 1

    original = df[log_col].copy()

    df.loc[outlier_dates, log_col] = np.nan
    df[log_col] = df[log_col].interpolate(method='time')
    print(f"Winsorized {len(outlier_dates)} values via time interpolation")

    fig_name = name.lower().replace(" ", "_")
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, original, color='lightgrey', linewidth=1.5, label='Original')
    plt.plot(df.index, df[log_col], color='steelblue', linewidth=1.5, label='Cleaned')
    plt.scatter(outlier_dates, original.loc[outlier_dates], color='red', zorder=5, label=f'Outliers ({len(outlier_dates)})')
    plt.title(f"Outlier Detection — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outliers_{fig_name}.png", bbox_inches='tight')
    plt.close()
    print(f"Outlier plot saved to 'outliers_{fig_name}.png'")

    return df

def add_fourier_features(df, period):
    if period == 52:
        t = df.index.isocalendar().week.astype(float)
    else:
        t = df.index.month.astype(float)
    df['fourier_sin_1'] = np.sin(2 * np.pi * t / period)
    df['fourier_cos_1'] = np.cos(2 * np.pi * t / period)
    return df

def analyze_distribution(series, name="Series"):
    print(f"\n--- Fitting Noise Distribution via scipy for: {name} ---")
    distributions = ['norm', 'expon', 'gamma', 'lognorm']
    results = []
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        params = dist.fit(series.values)
        sse = np.sum((dist.pdf(np.sort(series.values), *params) -
                      np.arange(1, len(series) + 1) / len(series)) ** 2)
        results.append((dist_name, sse, params))
    results.sort(key=lambda x: x[1])
    print("Top fitting distributions based on Sum-of-Squares Error:")
    for dist_name, sse, params in results[:3]:
        print(f"  {dist_name}: SSE={sse:.6f}, params={params}")
    return results

def main():
    print("=== RUNNING DATA PREPROCESSING & EDA  ===")
    
    monthly_series = load_and_aggregate_data(config.MONTHLY_DATA_PATH, "month")
    weekly_series = load_and_aggregate_data(config.WEEKLY_DATA_PATH, "week")
    
    print(f"Aggregated monthly observations: {monthly_series.shape[0]}")
    print(f"Aggregated weekly observations: {weekly_series.shape[0]}")
    
    print("\n[Descriptive Statistics] Monthly Target Column Summary:")
    print(monthly_series[config.TARGET_COL].describe())
    
    analyze_distribution(monthly_series[config.TARGET_COL], "Raw Monthly Leads")
    
    print("\n[Decomposition] Decomposing monthly target series...")
    decomposition = seasonal_decompose(monthly_series[config.TARGET_COL], model='additive', period=12)
    
    plt.figure(figsize=(10, 6))
    decomposition.plot()
    plt.savefig("monthly_decomposition.png", bbox_inches='tight')
    plt.close()
    print("Decomposition plot saved successfully to 'monthly_decomposition.png'")
    
    check_stationarity(monthly_series[config.TARGET_COL], "Raw Monthly Leads")
    
    print("\n Applying log transformation: log(y + 1)...")
    monthly_series = apply_log_transform(monthly_series)
    weekly_series = apply_log_transform(weekly_series)

    monthly_series = detect_and_handle_outliers(monthly_series, 'log_target', period=12, name="Monthly Log Target")
    weekly_series = detect_and_handle_outliers(weekly_series, 'log_target', period=52, name="Weekly Log Target")

    monthly_series = add_fourier_features(monthly_series, period=12)
    weekly_series = add_fourier_features(weekly_series, period=52)

    check_stationarity(monthly_series['log_target'], "Log-transformed Monthly Leads")

    monthly_series.to_csv("processed_monthly_global.csv")
    weekly_series.to_csv("processed_weekly_global.csv")
    print("\nProcessed outputs saved as 'processed_monthly_global.csv' and 'processed_weekly_global.csv'")

if __name__ == "__main__":
    main()