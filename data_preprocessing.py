import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from fitter import Fitter
import config

def load_and_aggregate_data(file_path, time_col):
    df = pd.read_csv(file_path)
    df[time_col] = pd.to_datetime(df[time_col])
    
    #Aggregating leads and spend globally across all segments
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

def analyze_distribution(series, name="Series"):
    print(f"\n--- Fitting Noise Distribution via Fitter for: {name} ---")
    #Using standard distributions
    f = Fitter(series.values, distributions=['norm', 'expon', 'gamma', 'lognorm'])
    f.fit()
    print("Top fitting distributions based on Sum-of-Squares Error:")
    print(f.summary().head(3))
    return f

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
    monthly_series['log_target'] = np.log1p(monthly_series[config.TARGET_COL])
    monthly_series['log_exog'] = np.log1p(monthly_series[config.EXOG_COL])
    
    weekly_series['log_target'] = np.log1p(weekly_series[config.TARGET_COL])
    weekly_series['log_exog'] = np.log1p(weekly_series[config.EXOG_COL])
    
    check_stationarity(monthly_series['log_target'], "Log-transformed Monthly Leads")
        
    monthly_series.to_csv("processed_monthly_global.csv")
    weekly_series.to_csv("processed_weekly_global.csv")
    print("\nProcessed outputs saved as 'processed_monthly_global.csv' and 'processed_weekly_global.csv'")

if __name__ == "__main__":
    main()