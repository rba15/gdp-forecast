"""
baseline_arima.py
-----------------
This script trains a simple ARIMA model to forecast GDP for one or more countries.
It uses only historical GDP data as input (no exogenous features) to establish a baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = "preprocessed_economic_data.parquet"  # From your preprocessing step
TARGET = "NY.GDP.MKTP.CD"                         # GDP indicator column
TEST_SIZE = 5                                     # Last N years for testing
COUNTRIES_TO_RUN = None                           # e.g. ["United States", "China"] or None for all
PLOT_RESULTS = True

# -----------------------------
# METRICS FUNCTION
# -----------------------------
def evaluate_forecast(y_true, y_pred):
    """Compute standard regression error metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# -----------------------------
# BASELINE ARIMA FORECAST
# -----------------------------
def run_arima_forecast(df, country):
    """Fit ARIMA model and forecast GDP for a single country."""
    df_country = df[df["country"] == country].sort_values("date")
    series = df_country[TARGET].dropna()

    if len(series) < 10:
        print(f"Skipping {country}: Not enough data ({len(series)} points)")
        return None

    # Train-test split (chronological)
    train, test = series[:-TEST_SIZE], series[-TEST_SIZE:]

    # Automatically select ARIMA order using AIC minimization
    try:
        model = sm.tsa.arima.model.ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
    except Exception as e:
        print(f"ARIMA failed for {country}: {e}")
        return None

    # Forecast
    forecast = model_fit.forecast(steps=TEST_SIZE)
    forecast.index = test.index

    # Evaluate
    rmse, mae, mape = evaluate_forecast(test, forecast)

    print(f"\nCountry: {country}")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

    # Plot
    if PLOT_RESULTS:
        plt.figure(figsize=(10,5))
        plt.plot(train.index, train, label="Train")
        plt.plot(test.index, test, label="Test", color='orange')
        plt.plot(forecast.index, forecast, label="Forecast", color='green')
        plt.title(f"GDP Forecast (ARIMA) - {country}")
        plt.xlabel("Year")
        plt.ylabel("GDP")
        plt.legend()
        plt.show()

    return {
        "country": country,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }

# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"])

    # Filter countries if specified
    countries = COUNTRIES_TO_RUN or df["country"].unique()

    results = []
    for country in countries:
        res = run_arima_forecast(df, country)
        if res:
            results.append(res)

    # Summarize results
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== Summary ===")
        print(results_df.describe())
        results_df.to_csv("baseline_arima_results.csv", index=False)
        print("Saved results to baseline_arima_results.csv")

if __name__ == "__main__":
    main()
