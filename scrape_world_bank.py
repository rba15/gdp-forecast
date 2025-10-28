import wbdata
import pandas as pd
import datetime
import time
import json
import os
import logging
from sqlalchemy import create_engine
import urllib

# -----------------------------
# CONFIGURATION
# -----------------------------

# Parquet file for storing intermediate/final data
PARQUET_FILE = "world_bank_data.parquet"

# Checkpoint file to resume scraping
CHECKPOINT_FILE = "fetched_countries.json"

# Batching configuration
BATCH_SIZE = 10         # Number of indicators per batch
RETRY_ATTEMPTS = 3      # Number of retry attempts for failed requests
RETRY_DELAY = 5         # Delay between retries in seconds

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraping.log"),
        logging.StreamHandler()
    ]
)

# Date range for World Bank data
DATA_DATE = (datetime.datetime(1960, 1, 1), datetime.datetime.today())

# -----------------------------
# FUNCTIONS
# -----------------------------

def load_db_config(file_path='db_config.json'):
    """Load database configuration from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_db_engine(db_config):
    """Create SQLAlchemy engine for MS SQL Server."""
    params = urllib.parse.quote_plus(
        f"DRIVER={db_config['driver']};"
        f"SERVER={db_config['server']};"
        f"DATABASE={db_config['database']};"
        f"UID={db_config['username']};"
        f"PWD={db_config['password']};"
        "TrustServerCertificate=yes"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def get_all_indicators():
    """Fetch all available indicators from World Bank API."""
    indicators = wbdata.get_indicator(display=False)
    indicator_dict = {i['id']: i['name'] for i in indicators}
    logging.info(f"Total indicators fetched: {len(indicator_dict)}")
    return indicator_dict

def load_checkpoint():
    """Load checkpoint to know which countries are already fetched."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(fetched_countries):
    """Save checkpoint after processing countries."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(fetched_countries), f)

def fetch_data_with_retry(country_code, indicators_subset):
    """
    Fetch data for a single country and batch of indicators.
    Retries in case of errors.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            df = wbdata.get_dataframe(indicators_subset, country=country_code, data_date=DATA_DATE)
            return df
        except Exception as e:
            logging.warning(f"Error fetching {country_code} (attempt {attempt+1}/{RETRY_ATTEMPTS}): {e}")
            time.sleep(RETRY_DELAY)
    logging.error(f"Failed to fetch data for {country_code} after {RETRY_ATTEMPTS} attempts")
    return pd.DataFrame()

def fetch_all_data():
    """
    Fetch all indicators for all countries with batching and checkpointing.
    Returns combined DataFrame in long format: country | year | indicator | value
    """
    indicators = get_all_indicators()
    countries = wbdata.get_country(display=False)
    fetched_countries = load_checkpoint()
    all_data = []

    for country in countries:
        country_code = country['id']
        country_name = country['name']

        # Skip if already fetched
        if country_code in fetched_countries:
            logging.info(f"Skipping {country_name}, already fetched.")
            continue

        logging.info(f"Fetching data for {country_name} ({country_code})")

        # Batch indicators
        indicator_keys = list(indicators.keys())
        for i in range(0, len(indicator_keys), BATCH_SIZE):
            batch_keys = indicator_keys[i:i+BATCH_SIZE]
            batch_dict = {k: indicators[k] for k in batch_keys}
            df = fetch_data_with_retry(country_code, batch_dict)

            if not df.empty:
                df.reset_index(inplace=True)  # Convert index (date) to column
                df_long = df.melt(id_vars=['date'], value_vars=batch_dict.keys(),
                                  var_name='indicator', value_name='value')
                df_long['country'] = country_name
                all_data.append(df_long)

        # Save checkpoint after each country
        fetched_countries.add(country_code)
        save_checkpoint(fetched_countries)

        # Save intermediate Parquet file
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_parquet(PARQUET_FILE, engine='pyarrow', index=False, compression='snappy')
            logging.info(f"Intermediate data saved to {PARQUET_FILE}")

    # Combine all batches
    if all_data:
        final_data = pd.concat(all_data, ignore_index=True)
        return final_data
    else:
        logging.warning("No data fetched.")
        return pd.DataFrame()

def load_parquet_to_sql(parquet_file, engine, table_name='world_bank_data'):
    """Load Parquet file into MS SQL table (long format)."""
    df = pd.read_parquet(parquet_file)
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    logging.info(f"Data loaded into SQL table '{table_name}'")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Load DB config and create engine
    db_config = load_db_config('db_config.json')
    engine = create_db_engine(db_config)

    # Fetch all data (or load from existing Parquet)
    if os.path.exists(PARQUET_FILE):
        logging.info(f"Loading existing Parquet file {PARQUET_FILE}")
        df = pd.read_parquet(PARQUET_FILE)
    else:
        df = fetch_all_data()
        if not df.empty:
            df.to_parquet(PARQUET_FILE, engine='pyarrow', index=False, compression='snappy')
            logging.info(f"Final data saved to {PARQUET_FILE}")

    # Load into MS SQL
    if not df.empty:
        load_parquet_to_sql(PARQUET_FILE, engine)