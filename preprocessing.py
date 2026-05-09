import pandas as pd
from sklearn.model_selection import train_test_split
from constants import TARGET_PRICE, CURRENT_PRICE


def load_data(filename: str):
    df = pd.read_csv(f"./datasets/{filename}")
    return df


def get_summary(df: pd.DataFrame):
    print(f"df.info(): {df.info()}")
    print()
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    print(f"DataFrame has {num_rows} rows and {num_cols} columns.")
    print()
    print(f"Summary stats: {df.describe()}")
    print()
    print(f"Number of NaNs: {df.isnull().sum()}")
    print()
    print(f"Data types: {df.dtypes}")
    print()
    print(f"df.head(): {df.head()}")


def prep_and_split_data(train_df):
    """
    Assumes the df has already been cleaned and feature engineered. Drops some cols and does
    train test split"""
    cols_to_drop = [
        "ID",
        "p41",
        "p42",
        "p43",
        "p44",
        "p45",
        "p46",
        "p47",
        "p48",
        "p49",
        "p50",
    ]

    X = train_df.drop(columns=cols_to_drop, errors="ignore")
    y_price = train_df[TARGET_PRICE]
    y_class = (
        train_df[TARGET_PRICE] < train_df[CURRENT_PRICE]
    )  # this says if it went down (0 or 1)

    # split for classification
    X_train, X_val, y_train_class, y_val_class = train_test_split(
        X, y_class, test_size=0.2, shuffle=False
    )

    # split for regression
    _, _, y_train_price, y_val_price = train_test_split(
        X, y_price, test_size=0.2, shuffle=False
    )

    return X_train, X_val, y_train_class, y_val_class, y_train_price, y_val_price
