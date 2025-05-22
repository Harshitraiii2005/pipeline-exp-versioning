import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# ==========================
# Logger Configuration
# ==========================

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)  # Use logging.DEBUG, not string 'DEBUG'

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Prevent duplicate logs in Jupyter environments
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_params(param_path: str) -> dict:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', param_path)
        return params
    except FileNotFoundError as e:
        logger.error('Parameter file not found: %s', param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Error loading parameters: %s', e)
        raise    



def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error parsing data from %s: %s', data_url, e)
        raise
    except Exception as e:
        logger.error('Error loading data from %s: %s', data_url, e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        if 'label_num' in df.columns:
            df.drop(columns=['label_num'], inplace=True)
        if 'label' in df.columns:
            df.rename(columns={'label': 'target'}, inplace=True)

        logger.debug('Data preprocessed successfully')
        return df
    except KeyError as e:
        logger.error('Error preprocessing data: %s', e)
        raise
    except Exception as e:
        logger.error('Error preprocessing data: %s', e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_dir = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)

        train_df.to_csv(os.path.join(raw_data_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data_dir, 'test.csv'), index=False)
        logger.debug('Data saved to %s', raw_data_dir)
    except Exception as e:
        logger.error('Error saving data to %s: %s', data_path, e)
        raise


def main():
    try:
        params= load_params(param_path='params.yaml')
        logger.info('Parameters loaded successfully.')
        test_size = params['data_ingestion']['test_size']
        data_path = 'experiments/spam_ham_dataset.csv'
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_df, test_df, data_path='./data')

        logger.info('Data ingestion pipeline completed successfully.')

    except Exception as e:
        logger.error('Error in main function: %s', e)
        raise

if __name__ == "__main__":
    main()
