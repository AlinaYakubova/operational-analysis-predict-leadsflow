import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import config

optuna.logging.set_verbosity(optuna.logging.WARNING)

ENRICHED_COLS = ['source_group', 'pipeline_group', 'course_group']

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def prepare_supervised_data(df, n_lags=4, extra_cols=None):
    df_feat = df.copy()
    for lag in range(1, n_lags + 1):
        df_feat[f'target_lag_{lag}'] = df_feat['log_target'].shift(lag)
    df_feat = df_feat.dropna()

    feature_cols = (
        [f'target_lag_{lag}' for lag in range(1, n_lags + 1)]
        + ['log_exog', 'outlier_flag', 'fourier_sin_1', 'fourier_cos_1']
        + (extra_cols or [])
    )
    X = df_feat[feature_cols].values
    y = df_feat['log_target'].values
    return X, y, df_feat, feature_cols

def run_optuna_study(X_train, y_train, horizon, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
        val_size = horizon
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=config.RANDOM_STATE, **params)
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(np.mean((y_val - preds) ** 2))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Validation RMSE:", study.best_value)
    print("Best params found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    return study.best_params

def fit_xgboost(X_train, y_train, params):
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=config.RANDOM_STATE, **params)
    model.fit(X_train, y_train, verbose=False)
    return model

def sliding_window_forecast(model, X_test, y_train, n_lags, horizon):
    forecast_log = []
    current_window = list(y_train[-n_lags:])

    for i in range(horizon):
        exog_features = list(X_test[i, n_lags:])
        input_features = np.array(current_window + exog_features).reshape(1, -1)
        next_pred = model.predict(input_features)[0]
        forecast_log.append(next_pred)

        current_window.pop(0)
        current_window.append(next_pred)

    return np.exp(forecast_log) - 1

def run_pipeline(df, n_lags, horizon, label, extra_cols=None):
    X, y, df_feat, _ = prepare_supervised_data(df, n_lags=n_lags, extra_cols=extra_cols)
    X_train, X_test = X[:-horizon], X[-horizon:]
    y_train = y[:-horizon]

    print(f"\n[Optimization] Optuna study — {label} ({X_train.shape[1]} features)...")
    best_params = run_optuna_study(X_train, y_train, horizon)

    model = fit_xgboost(X_train, y_train, best_params)

    print(f"[Forecasting] Sliding forecast — {label}...")
    forecast = sliding_window_forecast(model, X_test, y_train, n_lags, horizon)

    return forecast, df_feat

def plot_forecast_comparison(df, df_feat, test_dates, actual_final, forecasts, titles, horizon,
                             save_path="xgboost_forecast_vs_actuals.png"):
    _, axes = plt.subplots(len(forecasts), 1, figsize=(10, 5 * len(forecasts)))

    for ax, forecast, title in zip(axes, forecasts, titles):
        ax.plot(df_feat.index[:-horizon], df.loc[df_feat.index[:-horizon], config.TARGET_COL],
                label="Train Actuals", color="blue")
        ax.plot(test_dates, actual_final, label="Test Actuals", color="cyan")
        ax.plot(test_dates, forecast, label="XGBoost Forecast", color="green", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Timeline")
        ax.set_ylabel("Leads Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Forecast comparison chart saved to '{save_path}'")

def main():
    print("=== RUNNING MACHINE LEARNING (XGBOOST + OPTUNA) ===")

    df_base = pd.read_csv("processed_weekly_global.csv", index_col=0, parse_dates=True)
    df_enriched = pd.read_csv("processed_weekly_enriched.csv", index_col=0, parse_dates=True)

    n_lags = 4
    horizon = config.WEEKLY_TEST_HORIZON

    forecast_base, df_feat = run_pipeline(df_base, n_lags, horizon, label="Base (8 features)")
    forecast_enriched, _ = run_pipeline(df_enriched, n_lags, horizon, label="Enriched (11 features)", extra_cols=ENRICHED_COLS)

    # RFE is skipped because there are only three new enriched features.

    test_dates = df_feat.index[-horizon:]
    actual_final = df_base.loc[test_dates, config.TARGET_COL].values

    for label, forecast in [("Base", forecast_base), ("Enriched", forecast_enriched)]:
        mae, rmse, mape = calculate_metrics(actual_final, forecast)
        print(f"\n=== XGBOOST {label.upper()} PERFORMANCE ===")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : {mape:.2f}%")

    pd.DataFrame({
        'Actual': actual_final,
        'XGBoost_Base': forecast_base,
        'XGBoost_Enriched': forecast_enriched
    }, index=test_dates).to_csv("predictions_xgboost.csv")

    plot_forecast_comparison(
        df_base, df_feat, test_dates, actual_final,
        forecasts=[forecast_base, forecast_enriched],
        titles=[
            "XGBoost — Base (lags + exog + fourier)",
            "XGBoost — Enriched (+ mapped source, pipeline, course)"
        ],
        horizon=horizon
    )

if __name__ == "__main__":
    main()
