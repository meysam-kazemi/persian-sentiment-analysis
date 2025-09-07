import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_config, read_df

def load_and_preprocess_data(config):
    """
    Loads raw data, cleans it, splits it into training and testing sets,
    and saves them to the processed data directory.
    """
    text_col = config.get('DF', 'text_col')
    label_col = config.get('DF', 'label_col')
    processed_path = config.get('DATA', 'processed_path')

    df = read_df(config)

    # Basic cleaning: Drop rows with missing values in text or label columns
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    
    # Ensure text column is of type string
    df[text_col] = df[text_col].astype(str)

    # Split the data into training and testing sets
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(config.get('MODEL', 'test_size')),
        random_state=int(config.get('MODEL', 'random_state')),
        stratify=y
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # Create processed data directory if it doesn't exist
    os.makedirs(processed_path, exist_ok=True)

    # Save the split data to new CSV files
    train_df = pd.DataFrame({text_col: X_train, label_col: y_train})
    test_df = pd.DataFrame({text_col: X_test, label_col: y_test})

    train_df.to_csv(processed_path+'/train.csv', index=False)
    test_df.to_csv(processed_path+'/test.csv', index=False)
    print("Data preprocessing finished successfully.")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    config = read_config()
    load_and_preprocess_data(config)
