from sklearn.metrics import accuracy_score


def train_and_evaluate_lgb_classifier(X_train, y_train, X_val, y_val, params=None):
    """
    Trains a LightGBM model with custom parameters, evaluates it,
    and returns the trained model and accuracy.
    """
    if params is None:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 7,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    print(f"Training with params: {params}")

    model = lgb.LGBMClassifier(**params)  # ** unpacks params
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return model, val_accuracy


import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def train_and_evaluate_lgb_regressor(
    X_train, y_train_continuous, X_val, y_val_continuous, params=None
):
    """
    Trains a LightGBM regressor to predict  prices/returns.
    """
    if params is None:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.025,
            "max_depth": 7,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    print(f"Training Regressor with params: {params}")

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train_continuous)

    # using mse to evaluate
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val_continuous, val_predictions)

    print(f"Validation MSE: {val_mse:.6f}")

    return model, val_mse
