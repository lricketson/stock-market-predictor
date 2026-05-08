import pandas as pd
from constants import TARGET_COL
from sklearn.model_selection import train_test_split


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


def find_nans_per_column(df: pd.DataFrame):
    cols_to_drop = []
    for col in df.columns:
        num_missing_vals = df[col].isnull().sum()
        proportion_missing_vals = num_missing_vals / len(df[col])
        print(
            f"Column {col} is {proportion_missing_vals * 100}% missing values (total count {num_missing_vals})."
        )
        if proportion_missing_vals >= 0.2:
            cols_to_drop.append(col)
    print("-" * 60)
    print(f"Cols recommended to drop: {[col for col in cols_to_drop]}")
    return cols_to_drop


def drop_nans(df: pd.DataFrame):
    """
    Takes in a dataframe and drops all rows with a NaN value.
    """
    return df.dropna()


# def fill_nans_df(df: pd.DataFrame, value):
#     """
#     Fills in every NaN in the whole DataFrame with the same value.
#     """
#     df = df.fillna(value)
#     return df


def fill_nans_column(df: pd.DataFrame, col_name: str, value):
    """
    Takes in a DataFrame and a column name, and fills all the NaNs in that column with the given value,
    then returns the newly imputed DataFrame.
    """
    df = df.copy()
    df[col_name] = df[col_name].fillna(value)
    return df


def drop_cols(df: pd.DataFrame, cols_list: list):
    """
    Takes in a DataFrame and a list of columns, drops those named columns from the DataFrame, and returns it.
    """
    df = df.copy()
    df = df.drop(columns=cols_list)
    return df


def forward_fill_prices(df: pd.DataFrame):
    """
    Fills missing values with the last known valid observation.
    Critical for financial time-series to prevent lookahead bias and price crashes.
    """
    df = df.copy()
    return df.ffill()


def merge_data_dfs():
    return


def prep_and_split_data(train_df):
    """
    This assumes the df has already been cleaned. It returns the training df split into validation data.
    """
    cols_to_drop = ["Date", TARGET_COL]  # Potentially add more
    X = train_df.drop(columns=cols_to_drop)
    y = train_df[TARGET_COL]

    # shuffle=False guarantees we train on the PAST and validate on the FUTURE.
    # If you shuffle, you will accidentally cheat, get 99% accuracy locally,
    # and fail the actual competition.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return X_train, X_val, y_train, y_val
