"""
train_xgboost_gdp.py
---------------------
This script trains and tunes an XGBoost model to nowcast GDP using economic indicators.
It uses the pre-split training and test datasets created earlier (to avoid data leakage),
performs feature selection, hyperparameter tuning, and final evaluation on the test set.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURATION
# -----------------------------
TRAIN_FILE = "train_data.parquet"
TEST_FILE = "test_data.parquet"
TARGET = "NY.GDP.MKTP.CD"   # GDP column name (World Bank indicator)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_gdp_model.joblib")

# -----------------------------
# METRICS FUNCTION
# -----------------------------
def evaluate_forecast(y_true, y_pred):
    """Compute regression performance metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2

# -----------------------------
# FEATURE SELECTION FUNCTION
# -----------------------------
def feature_selection_xgboost(X, y, top_n=20):
    """Use XGBoost feature importance to select top N features."""
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"\nSelected Top {top_n} Features:")
    print(top_features)
    return top_features

# -----------------------------
# MAIN TRAINING PIPELINE
# -----------------------------
def main():
    print("Loading train and test datasets...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    # Sort chronologically to maintain forecasting logic
    train_df = train_df.sort_values(["country", "date"])
    test_df = test_df.sort_values(["country", "date"])

    # Drop any remaining NaNs in GDP
    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])

    # Separate features and target
    features = [col for col in train_df.columns if col not in ["country", "date", TARGET]]
    X_train_full = train_df[features]
    y_train_full = train_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]

    # Split train into train/validation (80/20 split, time-based)
    split_index = int(len(X_train_full) * 0.8)
    X_train, X_val = X_train_full.iloc[:split_index], X_train_full.iloc[split_index:]
    y_train, y_val = y_train_full.iloc[:split_index], y_train_full.iloc[split_index:]

    print(f"\nTrain size: {len(X_train)} | Validation size: {len(X_val)} | Test size: {len(X_test)}")

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    top_features = feature_selection_xgboost(X_train, y_train, top_n=25)
    X_train = X_train[top_features]
    X_val = X_val[top_features]
    X_test = X_test[top_features]

    # -----------------------------
    # HYPERPARAMETER TUNING
    # -----------------------------
    print("\nRunning hyperparameter tuning with TimeSeriesSplit...")
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 2]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBRegressor(random_state=42, objective="reg:squarederror")
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Best CV RMSE: {-grid.best_score_:.4f}")

    # -----------------------------
    # FINAL MODEL TRAINING
    # -----------------------------
    best_model = grid.best_estimator_
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False
    )

    # -----------------------------
    # EVALUATE ON TEST SET
    # -----------------------------
    y_pred = best_model.predict(X_test)
    rmse, mae, mape, r2 = evaluate_forecast(y_test, y_pred)

    print("\n=== FINAL NOWCASTING RESULTS (on test set) ===")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
