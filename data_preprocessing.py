import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import logging
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_PARQUET_FILE = "world_bank_data.parquet"
PROCESSED_FILE = "world_bank_data_gdp_ready.parquet"

COL_MISSING_THRESHOLD = 0.3  # Drop columns with >30% missing
ROW_MISSING_THRESHOLD = 0.5  # Drop rows with >50% missing
CORR_THRESHOLD = 0.95        # Drop features >0.95 correlated
TARGET = 'NY.GDP.MKTP.CD'    # Target variable (GDP)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_gdp.log"),
        logging.StreamHandler()
    ]
)

# -----------------------------
# LOAD RAW DATA
# -----------------------------
if not os.path.exists(RAW_PARQUET_FILE):
    raise FileNotFoundError(f"{RAW_PARQUET_FILE} not found. Run the scraping script first.")

logging.info(f"Loading raw data from {RAW_PARQUET_FILE}")
df = pd.read_parquet(RAW_PARQUET_FILE)

# -----------------------------
# PIVOT LONG DATA TO WIDE FORMAT
# -----------------------------
logging.info("Pivoting data to wide format")
df_wide = df.pivot_table(index=['country', 'date'], columns='indicator', values='value')
df_wide.reset_index(inplace=True)

# -----------------------------
# HANDLE MISSING DATA
# -----------------------------
logging.info("Handling missing data")

# Drop columns with too many missing values
col_missing = df_wide.isnull().mean()
cols_to_drop = col_missing[col_missing > COL_MISSING_THRESHOLD].index
df_wide.drop(columns=cols_to_drop, inplace=True)
logging.info(f"Dropped columns with >{COL_MISSING_THRESHOLD*100}% missing: {list(cols_to_drop)}")

# Drop rows with too many missing values
row_missing = df_wide.isnull().mean(axis=1)
rows_to_drop = df_wide.index[row_missing > ROW_MISSING_THRESHOLD]
df_wide.drop(index=rows_to_drop, inplace=True)
logging.info(f"Dropped {len(rows_to_drop)} rows with >{ROW_MISSING_THRESHOLD*100}% missing values")

# Interpolate numeric columns per country
numeric_cols = df_wide.select_dtypes(include=[np.number]).columns
df_wide[numeric_cols] = df_wide.groupby('country')[numeric_cols].apply(lambda group: group.interpolate(method='linear', limit_direction='both'))

# Fill remaining numeric missing values with median
df_wide[numeric_cols] = df_wide[numeric_cols].fillna(df_wide[numeric_cols].median())

# -----------------------------
# DROP LEAKY GDP FEATURES
# -----------------------------
logging.info("Dropping leak-prone GDP features")

# Drop columns containing 'gdp' except target
gdp_cols = [col for col in df_wide.columns if 'gdp' in col.lower() and col != TARGET]
df_wide.drop(columns=gdp_cols, inplace=True)
logging.info(f"Dropped columns containing 'GDP': {gdp_cols}")

# Drop columns highly correlated (>0.95) with target GDP
numeric_cols = df_wide.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c != TARGET]
corr_with_target = df_wide[numeric_cols + [TARGET]].corr()[TARGET]
high_corr_cols = corr_with_target[abs(corr_with_target) > CORR_THRESHOLD].index.tolist()
high_corr_cols = [c for c in high_corr_cols if c != TARGET]
df_wide.drop(columns=high_corr_cols, inplace=True)
logging.info(f"Dropped columns highly correlated (>0.95) with GDP: {high_corr_cols}")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
logging.info("Feature engineering")

# Lagged GDP features (safe)
df_wide['GDP_lag1'] = df_wide.groupby('country')[TARGET].shift(1)
df_wide['GDP_lag2'] = df_wide.groupby('country')[TARGET].shift(2)

# Safe trade features
if 'NE.EXP.GNFS.CD' in df_wide.columns and 'NE.IMP.GNFS.CD' in df_wide.columns:
    df_wide['Trade_balance'] = df_wide['NE.EXP.GNFS.CD'] - df_wide['NE.IMP.GNFS.CD']
    df_wide['Exports_to_Imports_ratio'] = df_wide['NE.EXP.GNFS.CD'] / df_wide['NE.IMP.GNFS.CD']

# Log-transform highly skewed numeric columns (excluding target)
for col in df_wide.select_dtypes(include=[np.number]).columns:
    if col != TARGET and (df_wide[col] > 0).all():
        skewness = df_wide[col].skew()
        if abs(skewness) > 2:
            df_wide[f'log_{col}'] = np.log(df_wide[col])
            logging.info(f"Log-transformed {col} due to high skew ({skewness:.2f})")

# -----------------------------
# FEATURE SELECTION
# -----------------------------
logging.info("Feature selection")

# Remove near-constant columns
numeric_cols = df_wide.select_dtypes(include=[np.number]).columns
selector = VarianceThreshold(threshold=0.0)
selector.fit(df_wide[numeric_cols])
constant_cols = [c for i, c in enumerate(numeric_cols) if not selector.variances_[i] > 0]
df_wide.drop(columns=constant_cols, inplace=True)
logging.info(f"Dropped near-constant columns: {constant_cols}")

# -----------------------------
# SCALING
# -----------------------------
logging.info("Scaling numeric features")
numeric_cols = df_wide.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c != TARGET]
scaler = StandardScaler()
df_wide[numeric_cols] = scaler.fit_transform(df_wide[numeric_cols])

# -----------------------------
# SAVE CLEANED DATA
# -----------------------------
logging.info(f"Saving cleaned data to {PROCESSED_FILE}")
df_wide.to_parquet(PROCESSED_FILE, engine='pyarrow', index=False, compression='snappy')

logging.info("Preprocessing complete. Dataset is ready for GDP forecasting.")
