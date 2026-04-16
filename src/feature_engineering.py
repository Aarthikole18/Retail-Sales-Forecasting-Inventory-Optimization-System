def create_features(df):
    # Lag features (past sales)
    df["lag_1"] = df["qty_sold"].shift(1)
    df["lag_2"] = df["qty_sold"].shift(2)

    # Rolling mean (trend)
    df["rolling_mean_3"] = df["qty_sold"].rolling(3).mean()

    # Day of week (pattern)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Remove null values
    df = df.dropna()

    return df