## Phase 1: The Parachute (11:00 AM - 11:30 AM)

[] Download data and load into eda.ipynb.

[] Run get_summary and check for missing values or weird data types.

[] CRITICAL: Check the format of test.csv. Does it have multiple stocks (Asset_ID)? Is it daily or minute data?

[] Run the raw data through get_baseline.ipynb.

[] Submit the baseline submission.csv to Kaggle. Get a score on the board.

## Phase 2: The Feature Factory (11:30 AM - 12:30 PM)

If you have extra time, this is where you spend it. Model iteration is mostly just finding better features.

[] Apply engineer_financial_features from premades.py.

[] Run show_top_features. What is LightGBM ignoring? Drop the useless features to make the laptop run faster.

[] ITERATE: Create new features based on the top performers. If SMA_20 is the best feature, try making an SMA_50 or SMA_100. If Daily_Range is highly ranked, calculate a 5-day rolling average of the Daily_Range.

## Phase 3: Squeezing the Model (12:30 PM - 1:15 PM)

Now you tweak the model itself.

[] Hyperparameter Tuning: Manually adjust LightGBM.

- Change learning_rate from 0.1 to 0.05 or 0.01 (smaller = slower to train but usually more accurate).

- Change n_estimators (number of trees) from 100 to 300 or 500.

- Change max_depth to 5 or 7 to prevent overfitting.

[] Thresholding: By default, .predict() assumes a 50% probability is a "Buy" (1). Use model.predict_proba(X_val) instead. Maybe you only want to buy if the model is 60% or 70% confident! This will drastically improve your Sharpe Ratio by keeping you in cash during uncertain times.

## Phase 4: Ensembling (1:15 PM - 1:45 PM)

If LightGBM is maxed out, combine it with something else.

[] Train a second, simpler model like RandomForestClassifier or LogisticRegression on the exact same features.

[] Average their predictions. If LightGBM thinks 80% chance of a rally, and Logistic Regression thinks 60%, your final ensemble prediction is 70%. Models make different mistakes, so blending them smooths out your returns.

## Phase 5: The Final Lockdown (1:45 PM - 2:00 PM)

[] Stop coding.

[] Choose your final submission on the Kaggle platform (Kaggle lets you check a box next to your 2 favorite submissions to be judged on the Private Leaderboard).

[] Verify your final notebook has zero lookahead bias so you pass the code review.
