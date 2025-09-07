import os
import sys
import pandas as pd
import string
import re
import hazm
from hazm import Normalizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_config

import pandas as pd
from sklearn.model_selection import train_test_split
import os

import logging
from src.utils import read_config, read_df




def load_and_preprocess_data(config):
    """
    Loads raw data, cleans it, splits it into training and testing sets,
    and saves them to the processed data directory.
    """
    df = read_df(config)

    # Basic cleaning: Drop rows with missing values in text or label columns
    df.dropna(subset=[config.get('DF', 'text_col'), config.get('DF', 'label_col')], inplace=True)
    logging.info(f"Data shape after dropping missing values: {df.shape}")
    
    # Ensure text column is of type string
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str)

    # Split the data into training and testing sets
    X = df[config.TEXT_COLUMN]
    y = df[config.LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logging.info(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # Create processed data directory if it doesn't exist
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)

    # Save the split data to new CSV files
    train_df = pd.DataFrame({config.TEXT_COLUMN: X_train, config.LABEL_COLUMN: y_train})
    test_df = pd.DataFrame({config.TEXT_COLUMN: X_test, config.LABEL_COLUMN: y_test})

    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    logging.info(f"Training data saved to {config.TRAIN_DATA_PATH}")
    logging.info(f"Testing data saved to {config.TEST_DATA_PATH}")
    logging.info("Data preprocessing finished successfully.")

if __name__ == '__main__':
    preprocess_data()
