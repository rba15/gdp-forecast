# Global GDP Forecasting with World Bank Data

This project builds a **comprehensive end-to-end pipeline** to scrape, clean, and forecast global GDP using **World Bank economic data**.  
It combines classical and modern machine learning models (ARIMA, XGBoost, LSTM) to create robust forecasting models across countries.

---

## Project Overview

### 1 Data Scraping
We collect **historical economic indicators** for all countries globally using the **World Bank API** (`wbdata`).

- Automatically fetches all available economic indicators (imports, exports, inflation, etc.)
- Handles retries, batching, and logging for stability
- Stores results in **Parquet files** for efficient analytics

Script: `scrape_world_bank.py`

---

### 2️ Data Preprocessing
Cleans and prepares the scraped data for machine learning and forecasting.

- Handles missing values, fills gaps, and removes low-variance or duplicate columns
- Drops columns too correlated (>0.95) with GDP to prevent target leakage
- Outputs a clean, structured dataset for modeling

Script: `preprocess_data.py`

---

### 3️ Train/Test Split for Forecasting
Performs **chronological splitting** (not random) to simulate real-world forecasting.

- Ensures the model never sees future data
- Saves resulting datasets as:
  - `train_data.parquet`
  - `test_data.parquet`

Script: `train_test_split_forecasting.py`

---

### 4️ Exploratory Data Analysis (EDA)
Analyzes the training data to understand relationships between indicators and GDP.

- Generates plots and correlation heatmaps
- Performs univariate and multivariate analysis
- Avoids test data to prevent data leakage

Notebook: `eda_feature_analysis.ipynb`

---

### 5️ Baseline Forecasting (ARIMA)
Provides a baseline **ARIMA model** to forecast GDP trends over time.

- Trains on historical GDP data
- Validates forecasts with RMSE, MAE, MAPE, and R²
- Serves as a benchmark for ML and LSTM models

Script: `baseline_arima.py`

---

### 6️ Machine Learning Forecast (XGBoost)
Builds an **XGBoost regression model** to predict GDP from all other economic indicators.

- Performs **feature selection** via XGBoost importance
- Uses **TimeSeriesSplit** for time-aware cross-validation
- Conducts **hyperparameter tuning** with `GridSearchCV`
- Evaluates on held-out test set (no leakage)
- Saves model to `models/xgboost_gdp_model.joblib`

Script: `train_xgboost_gdp.py`

---

### 7️ Deep Learning Forecast (LSTM)
Final forecasting model using **Long Short-Term Memory (LSTM)** neural networks.

- Performs scaling, sequence generation, and feature filtering
- Tunes LSTM units, dropout, and learning rate
- Uses early stopping and model checkpointing
- Evaluates final model on test set
- Saves model and scalers for inference

Script: `train_lstm_gdp.py`

---

## Key Features

- Full end-to-end reproducible GDP forecasting pipeline  
- Time-series–aware train/test/validation handling (no leakage)  
- Feature selection and correlation analysis  
- Classical and deep learning models (ARIMA, XGBoost, LSTM)  
- Hyperparameter tuning and model persistence  
- Modular, well-commented, and production-ready scripts  

---

## Setup Instructions

### 1️ Clone the repository

git clone https://github.com/YOUR_USERNAME/worldbank-gdp-forecasting.git
cd worldbank-gdp-forecasting

### 2️ Create a virtual environment

python -m venv venv
source venv/bin/activate  # on macOS/Linux
venv\Scripts\activate     # on Windows

### 3️ Install dependencies

pip install -r requirements.txt

### 4️ Run the pipeline

python scrape_world_bank.py
python preprocess_data.py
python train_test_split_forecasting.py
python train_xgboost_gdp.py
python train_lstm_gdp.py

## Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R2)
