import pandas as pd
import numpy as np
from constants import TARGET_COL


def engineer_financial_features(raw_df):
    """
    Takes raw OHLCV data and generates machine-learning-ready features.
    Designed purely with functions for rapid hackathon iteration.
    """
    # Create a copy so we don't accidentally modify the original raw data
    df = raw_df.copy()

    # --- 1. CORE PRICE ACTION ---
    # Log returns: How much did the price change percentage-wise today?
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Daily Range: How wild was the trading day?
    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Open"]

    # --- 2. MOMENTUM (Moving Averages) ---
    # 5-day and 20-day simple moving averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Distance to SMA: Is the current price overextended?
    df["Dist_to_SMA_5"] = df["Close"] / df["SMA_5"] - 1
    df["Dist_to_SMA_20"] = df["Close"] / df["SMA_20"] - 1

    # --- 3. VOLATILITY (Risk) ---
    # Rolling standard deviation of returns
    df["Vol_5d"] = df["Log_Return"].rolling(window=5).std()
    df["Vol_20d"] = df["Log_Return"].rolling(window=20).std()

    # --- 4. LAGGED FEATURES (Memory) ---
    # "What happened yesterday?" ML models need to be explicitly told this.
    df["Return_Lag_1"] = df["Log_Return"].shift(1)
    df["Return_Lag_2"] = df["Log_Return"].shift(2)
    df["Vol_Lag_1"] = df["Vol_5d"].shift(1)

    # --- 5. THE LOOKAHEAD BIAS SHIFT (CRITICAL) ---
    # We must shift all newly created features down by 1 row.
    # This ensures 'today's' row only contains data known as of 'yesterday's' close.
    # We do NOT shift Open, High, Low, Close, Volume, Date, or Target.
    core_cols = ["Date", "Open", "High", "Low", "Close", "Volume", TARGET_COL]

    # Safely identify only the newly engineered columns
    feature_cols = [col for col in df.columns if col not in core_cols]

    # Shift them forward
    df[feature_cols] = df[feature_cols].shift(1)

    # Drop any rows that now have NaNs because of rolling windows or shifting
    df = df.dropna().reset_index(drop=True)

    return df


def evaluate_strategy_performance(val_df, predictions):
    """
    Simulates portfolio returns based on your model's predictions.
    Assumes a Long-Only strategy (1 = Buy/Hold, 0 = Sell/Go to Cash).
    """
    df = val_df.copy()
    df["Prediction"] = predictions

    # If the model predicts 1, you get that day's log return.
    # If the model predicts 0, you sit in cash (return = 0).
    df["Strategy_Return"] = df["Prediction"] * df["Log_Return"]

    # Calculate total percentage returns using exponentials (undoing the log)
    market_return = (np.exp(df["Log_Return"].sum()) - 1) * 100
    strategy_return = (np.exp(df["Strategy_Return"].sum()) - 1) * 100

    # Calculate Annualized Sharpe Ratio
    # (Daily return mean / Daily return std) * sqrt(252 trading days)
    risk_free_rate = 0.0001  # Rough daily assumption
    excess_returns = df["Strategy_Return"] - risk_free_rate

    if excess_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    print("\n=== VALIDATION PORTFOLIO PERFORMANCE ===")
    print(f"Market 'Buy & Hold' Return: {market_return:.2f}%")
    print(f"Your Model's Portfolio:     {strategy_return:.2f}%")
    print(f"Model Sharpe Ratio:         {sharpe_ratio:.2f}")

    return df


def show_top_features(trained_model, feature_names, top_n=10):
    """
    Extracts and ranks the features LightGBM found most useful.
    """
    importance = trained_model.feature_importances_

    # Create a clean dataframe of features and their scores
    feature_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values(by="Importance", ascending=False)

    print(f"\n=== TOP {top_n} STRONGEST FEATURES ===")
    print(feature_df.head(top_n).to_string(index=False))

    return feature_df


import lightgbm as lgb
from sklearn.metrics import accuracy_score


def train_and_evaluate_lgb(X_train, y_train, X_val, y_val, params=None):
    """
    Trains a LightGBM model with custom parameters, evaluates it,
    and returns the trained model and accuracy.
    """
    # Default parameters if none are provided
    if params is None:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    print(f"Training with params: {params}")

    # Initialize and train
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return model, val_accuracy


import numpy as np
import pandas as pd


def evaluate_strategy_performance(val_df, predictions):
    """
    Simulates portfolio returns based on your model's predictions.
    Assumes a Long-Only strategy (1 = Buy/Hold, 0 = Sell/Go to Cash).
    """
    df = val_df.copy()
    df["Prediction"] = predictions

    # If the model predicts 1, you get that day's log return.
    # If the model predicts 0, you sit in cash (return = 0).
    df["Strategy_Return"] = df["Prediction"] * df["Log_Return"]

    # Calculate total percentage returns using exponentials (undoing the log)
    market_return = (np.exp(df["Log_Return"].sum()) - 1) * 100
    strategy_return = (np.exp(df["Strategy_Return"].sum()) - 1) * 100

    # Calculate Annualized Sharpe Ratio
    # (Daily return mean / Daily return std) * sqrt(252 trading days)
    risk_free_rate = 0.0001  # Rough daily assumption
    excess_returns = df["Strategy_Return"] - risk_free_rate

    if excess_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    print("\n=== VALIDATION PORTFOLIO PERFORMANCE ===")
    print(f"Market 'Buy & Hold' Return: {market_return:.2f}%")
    print(f"Your Model's Portfolio:     {strategy_return:.2f}%")
    print(f"Model Sharpe Ratio:         {sharpe_ratio:.2f}")

    return df
