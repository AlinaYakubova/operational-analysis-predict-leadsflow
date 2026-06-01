import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import config

# Disable optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def prepare_supervised_data(df, n_lags=4):
    """Transforms time series into a supervised learning matrix (Lecture 8)."""
    df_feat = df.copy()
    
    # Generate target lags
    for lag in range(1, n_lags + 1):
        df_feat[f'target_lag_{lag}'] = df_feat['log_target'].shift(lag)
        
    # Drop rows with NaN caused by shifting
    df_feat = df_feat.dropna()
    
    # Define feature space (Lags + Exogenous Marketing Spend)
    feature_cols = [f'target_lag_{lag}' for lag in range(1, n_lags + 1)] + ['log_exog']
    
    X = df_feat[feature_cols].values
    y = df_feat['log_target'].values
    return X, y, df_feat

def main():
    print("=== RUNNING MACHINE LEARNING (XGBOOST + OPTUNA) ===")
    
    df = pd.read_csv("processed_weekly_global.csv", index_col=0, parse_dates=True)
    
    # Transform series to supervised matrix
    n_lags = 4
    X, y, df_feat = prepare_supervised_data(df, n_lags=n_lags)
    
    # Train-Test Split based on weekly horizon
    horizon = config.WEEKLY_TEST_HORIZON
    X_train, X_test = X[:-horizon], X[-horizon:]
    y_train, y_test = y[:-horizon], y[-horizon:]
    
    print(f"Train matrix size: {X_train.shape}, Test matrix size: {X_test.shape}")
    
    # Hyperparameter Tuning via Optuna
    print("\n[Optimization] Launching Optuna Study for XGBoost parameter fitting...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
        
        # Intermediate validation split within training data to avoid test leakage
        val_size = horizon
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.RANDOM_STATE,
            **params
        )
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(np.mean((y_val - preds) ** 2)) # Optimize based on Validation RMSE
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print("Best Validation RMSE:", study.best_value)
    print("Best params found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Fit Final Model on the whole training set
    best_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=config.RANDOM_STATE,
        **study.best_params
    )
    best_model.fit(X_train, y_train, verbose=False)
    
    # Sliding Window Multi-Step Forecast
    print(f"\n[Forecasting] Generating rolling sliding forecast for {horizon} weeks...")
    
    forecast_log = []
    # Initialize the rolling window with the last available training lags
    current_window = list(y_train[-n_lags:])
    
    for i in range(horizon):
        # Extract the exogenous marketing spend for the current test step
        current_exog = X_test[i, -1] 
        
        # Construct the current feature vector: [lag_4, lag_3, lag_2, lag_1, exog]
        input_features = np.array(current_window + [current_exog]).reshape(1, -1)
        
        # Predict next log value
        next_pred = best_model.predict(input_features)[0]
        forecast_log.append(next_pred)
        
        # Update rolling window: drop oldest lag, append new prediction
        current_window.pop(0)
        current_window.append(next_pred)
        
    # Inverse transformation and Evaluation
    forecast_final = np.exp(forecast_log) - 1
    
    # Get original scale actuals from the underlying dataframe
    test_dates = df_feat.index[-horizon:]
    actual_final = df.loc[test_dates, config.TARGET_COL].values
    
    # Save predictions for final comparison script
    predictions_df = pd.DataFrame({
        'Actual': actual_final,
        'XGBoost_Forecast': forecast_final
    }, index=test_dates)
    predictions_df.to_csv("predictions_xgboost.csv")
    
    mae, rmse, mape = calculate_metrics(actual_final, forecast_final)
    print("\n=== XGBOOST TEST PERFORMANCE ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")
    
    # Visual chart
    plt.figure(figsize=(10, 5))
    plt.plot(df_feat.index[:-horizon], df.loc[df_feat.index[:-horizon], config.TARGET_COL], label="Train Actuals", color="blue")
    plt.plot(test_dates, actual_final, label="Test Actuals", color="cyan")
    plt.plot(test_dates, forecast_final, label="XGBoost Sliding Forecast", color="green", linestyle="--")
    plt.title("Kodland Lead Volume Forecasting - XGBoost + Optuna")
    plt.xlabel("Timeline")
    plt.ylabel("Leads Count")
    plt.legend()
    plt.savefig("xgboost_forecast_vs_actuals.png", bbox_inches='tight')
    plt.close()
    print("Forecast visual chart saved to 'xgboost_forecast_vs_actuals.png'")

if __name__ == "__main__":
    main()