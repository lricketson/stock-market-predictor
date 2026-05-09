import pandas as pd


def show_top_features(trained_model, feature_names, top_n=10):
    """
    Extracts and ranks the features lgbm found most useful.
    """
    importance = trained_model.feature_importances_

    feature_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values(by="Importance", ascending=False)

    print(f"\nTOP {top_n} STRONGEST FEATURES")
    print(feature_df.head(top_n).to_string(index=False))

    return feature_df


def show_bottom_features(trained_model, feature_names, top_n=10):
    """
    Extracts and ranks the features lgbm found least useful.
    """
    importance = trained_model.feature_importances_

    feature_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values(by="Importance", ascending=True)

    print(f"\nTOP {top_n} WEAKEST FEATURES")
    print(feature_df.head(top_n).to_string(index=False))

    return feature_df
