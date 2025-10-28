"""
train_lstm_gdp.py
---------------------
Final GDP forecasting model using LSTM.
This script performs:
- Feature selection and scaling
- Sequence generation for supervised learning
- Hyperparameter tuning
- Model training with early stopping and validation
- Evaluation on test set
- Model saving for future inference
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# CONFIGURATION
# -----------------------------
TRAIN_FILE = "train_data.parquet"
TEST_FILE = "test_data.parquet"
TARGET = "NY.GDP.MKTP.CD"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_gdp_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scalers.joblib")

TIME_STEPS = 5  # Number of lagged time steps to use for prediction
EPOCHS = 100
BATCH_SIZE = 32

# -----------------------------
# METRICS FUNCTION
# -----------------------------
def evaluate_forecast(y_true, y_pred):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2

# -----------------------------
# SEQUENCE GENERATION FUNCTION
# -----------------------------
def create_sequences(X, y, time_steps=1):
    """Transform data into time series sequences for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# -----------------------------
# MAIN TRAINING PIPELINE
# -----------------------------
def main():
    print("Loading train and test datasets...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    # Sort data by time for consistency
    train_df = train_df.sort_values(["country", "date"])
    test_df = test_df.sort_values(["country", "date"])

    # Drop missing target values
    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])

    # -----------------------------
    # FEATURE SELECTION (simple + safe)
    # -----------------------------
    print("\nPerforming feature selection...")
    features = [col for col in train_df.columns if col not in ["country", "date", TARGET]]

    # Drop constant or quasi-constant features
    nunique = train_df[features].nunique()
    features = [f for f in features if nunique[f] > 1]

    # Drop features highly correlated (> 0.95) with target or each other
    corr = train_df[features + [TARGET]].corr().abs()
    high_corr = corr[TARGET][corr[TARGET] > 0.95].index.tolist()
    features = [f for f in features if f not in high_corr and "gdp" not in f.lower()]

    print(f"Selected {len(features)} features after filtering.")

    # -----------------------------
    # SCALING
    # -----------------------------
    print("\nScaling features...")
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_full = feature_scaler.fit_transform(train_df[features])
    y_train_full = target_scaler.fit_transform(train_df[[TARGET]])
    X_test_full = feature_scaler.transform(test_df[features])
    y_test_full = target_scaler.transform(test_df[[TARGET]])

    # -----------------------------
    # SPLIT TRAIN INTO TRAIN/VAL
    # -----------------------------
    split_index = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_index]
    y_train = y_train_full[:split_index]
    X_val = X_train_full[split_index:]
    y_val = y_train_full[split_index:]

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test_full)}")

    # -----------------------------
    # SEQUENCE GENERATION
    # -----------------------------
    print("\nCreating sequences for LSTM...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_full, y_test_full, TIME_STEPS)

    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"X_val_seq shape: {X_val_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")

    # -----------------------------
    # MODEL DEFINITION
    # -----------------------------
    def build_model(units=64, dropout=0.2, learning_rate=0.001):
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(TIME_STEPS, X_train_seq.shape[2])),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    # -----------------------------
    # HYPERPARAMETER TUNING (manual grid)
    # -----------------------------
    param_grid = {
        "units": [32, 64],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005]
    }

    best_rmse = float("inf")
    best_params = None
    best_model = None

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nTuning hyperparameters...")
    for units in param_grid["units"]:
        for dropout in param_grid["dropout"]:
            for lr in param_grid["learning_rate"]:
                print(f"\nTraining model with units={units}, dropout={dropout}, lr={lr}")
                model = build_model(units, dropout, lr)

                checkpoint_path = os.path.join(MODEL_DIR, "temp_best_lstm.keras")
                callbacks = [
                    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
                ]

                history = model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=0
                )

                val_loss = min(history.history["val_loss"])
                rmse = np.sqrt(val_loss)
                print(f"Validation RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (units, dropout, lr)
                    best_model = model

    print(f"\n✅ Best LSTM params: units={best_params[0]}, dropout={best_params[1]}, lr={best_params[2]}")
    print(f"Best Validation RMSE: {best_rmse:.4f}")

    # -----------------------------
    # FINAL EVALUATION
    # -----------------------------
    y_pred_scaled = best_model.predict(X_test_seq)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test_seq)

    rmse, mae, mape, r2 = evaluate_forecast(y_true, y_pred)
    print("\n=== FINAL LSTM NOWCASTING RESULTS (on test set) ===")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    # -----------------------------
    # SAVE FINAL MODEL & SCALERS
    # -----------------------------
    best_model.save(MODEL_PATH)
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, SCALER_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Scalers saved to {SCALER_PATH}")

if __name__ == "__main__":
    main()
