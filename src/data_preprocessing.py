import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def clean_data(df):
    # Remove missing values
    df = df.dropna()

    # Remove negative values
    df = df[df["qty_sold"] >= 0]

    # Sort by date
    df = df.sort_values("date")

    return df