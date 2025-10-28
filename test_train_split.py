import pandas as pd
import numpy as np
import logging
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
CLEANED_FILE = "world_bank_data_gdp_ready.parquet"
TRAIN_FILE = "gdp_train.parquet"
TEST_FILE = "gdp_test.parquet"

TARGET = 'NY.GDP.MKTP.CD'
TEST_SIZE = 0.2  # Fraction of data to use as test (latest years)
LAG_FEATURES = [col for col in pd.read_parquet(CLEANED_FILE).columns if 'lag' in col]  # lag features already created

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_test_split.log"),
        logging.StreamHandler()
    ]
)

# -----------------------------
# LOAD CLEANED DATA
# -----------------------------
if not os.path.exists(CLEANED_FILE):
    raise FileNotFoundError(f"{CLEANED_FILE} not found. Run preprocessing first.")

logging.info(f"Loading cleaned data from {CLEANED_FILE}")
df = pd.read_parquet(CLEANED_FILE)

# -----------------------------
# SORT DATA
# -----------------------------
logging.info("Sorting data by country and date")
df.sort_values(by=['country', 'date'], inplace=True)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
logging.info("Performing time-series aware train/test split by country")

train_list = []
test_list = []

for country, group in df.groupby('country'):
    n = len(group)
    test_n = int(n * TEST_SIZE)
    train_n = n - test_n

    train_group = group.iloc[:train_n]
    test_group = group.iloc[train_n:]

    train_list.append(train_group)
    test_list.append(test_group)

train_df = pd.concat(train_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)

logging.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# -----------------------------
# DROP ROWS WITH NAN IN TARGET
# -----------------------------
train_df = train_df.dropna(subset=[TARGET])
test_df = test_df.dropna(subset=[TARGET])

# -----------------------------
# SAVE TRAIN / TEST
# -----------------------------
train_df.to_parquet(TRAIN_FILE, engine='pyarrow', index=False, compression='snappy')
test_df.to_parquet(TEST_FILE, engine='pyarrow', index=False, compression='snappy')

logging.info(f"Train and test datasets saved as {TRAIN_FILE} and {TEST_FILE}")

# -----------------------------
# OPTIONAL: Prepare LSTM 3D input (samples, timesteps, features)
# -----------------------------
def create_lstm_sequences(df, features, target, timesteps=3):
    """
    Convert dataframe into LSTM-friendly sequences.
    Returns X (samples, timesteps, features), y (samples,)
    """
    X_list = []
    y_list = []
    
    for country, group in df.groupby('country'):
        group = group.sort_values('date')
        data = group[features].values
        target_data = group[target].values
        for i in range(timesteps, len(group)):
            X_list.append(data[i-timesteps:i, :])
            y_list.append(target_data[i])
    
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# Example usage for later LSTM training
FEATURES = [c for c in df.columns if c not in ['country', 'date', TARGET]]
# X_train, y_train = create_lstm_sequences(train_df, FEATURES, TARGET)
# X_test, y_test = create_lstm_sequences(test_df, FEATURES, TARGET)

logging.info("LSTM sequences can be created using `create_lstm_sequences` function")
