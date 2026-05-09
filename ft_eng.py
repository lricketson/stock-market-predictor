from constants import CURRENT_PRICE


def engineer_features(df):
    df = df.copy()
    # recent value ratios
    df["Last_5_ratio"] = df[CURRENT_PRICE] / df["p35"] - 1
    df["Last_10_ratio"] = df[CURRENT_PRICE] / df["p30"] - 1

    recent_10_cols = [f"p{i}" for i in range(30, 40)] + [CURRENT_PRICE]
    recent_5_cols = [f"p{i}" for i in range(35, 40)] + [CURRENT_PRICE]
    # stds
    df["Std_10"] = df[recent_10_cols].std(axis=1)
    df["Std_5"] = df[recent_5_cols].std(axis=1)

    df["overall_change_r"] = df[CURRENT_PRICE] / df["p1"]
    df["halfway_change_r"] = df[CURRENT_PRICE] / df["p20"]

    return df
