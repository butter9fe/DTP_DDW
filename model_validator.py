from library import get_features_targets, split_data, predict_linreg, r2_score, mean_squared_error, build_model_with_validation
from constants import ALL_FEATURES, TARGET, FILE_NAME
import numpy as np
import pandas as pd

df: pd.DataFrame = pd.read_csv(FILE_NAME)

# (1) Extract the features and the target
df_features, df_target = get_features_targets(df, ALL_FEATURES, TARGET)

# (2) Split the data set into training and test
data = split_data(df_features, df_target, random_state=100)

# (3) Build model
model, best_features = build_model_with_validation(data, ALL_FEATURES)

# (4) Now try predicting with [Test] dataset!
pred = predict_linreg(data["test_features"][best_features].to_numpy(), model["beta"], model["means"], model["stds"])
target: np.ndarray = data["test_target"].to_numpy()

# Calculate metrics
r2: float = r2_score(target, pred)
mse: float = mean_squared_error(target, pred)

print(f"{best_features}: R2: {r2} | MSE: {mse}")