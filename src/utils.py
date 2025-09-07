import os
import pandas as pd
from configparser import ConfigParser

def read_config(config_path='config.ini'):
    if not os.path.exists(config_path):
        print('[Warning] Config file not exists.')
        return None
    config = ConfigParser()
    config.read(config_path)
    return config

def read_df(config):
    df = pd.read_csv(
        config.get('DATA', 'path'),
        encoding=config.get('DATA', 'encoding'),
        sep=config.get('DATA', 'sep'),
        on_bad_lines=config.get('DATA', 'on_bad_lines'),
        engine=config.get('DATA', 'engine')
        )
    return df
