- Strategies for a column having missing values:
- <5% missing: Impute (median for numeric, mode for categorical)
- 5-20% missing: drop column or investigate pattern
- 20%+ missing: drop column

- tree based methods (like LightGBM) are robust against outliers, and the mode needs to see them to learn, so don't drop.

for LightGBM hyperparameters:

- max_depth should be between 3 and 7
- for learning_rate, smaller is usually safer. combine it with a slightly higher n_estimators
