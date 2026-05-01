import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found!")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty!")

    return df