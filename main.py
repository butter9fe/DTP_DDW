from library import get_features_targets, split_data, build_model_linreg, predict_linreg, r2_score, mean_squared_error
from constants import ALL_FEATURES, CLEANED_FEATURES, TARGET
import numpy as np
import pandas as pd

df: pd.DataFrame = pd.read_csv("Test_3.csv")

# All features in string, followed by just the features we're using for our model in an array
# So this will look like [str, str, ..., list]
features: list= ALL_FEATURES + [CLEANED_FEATURES]
for index, feature in enumerate(features):
    # Extract the features and the target
    df_features, df_target = get_features_targets(df, feature, TARGET)

    # Split the data set into training and test
    data = split_data(df_features, df_target)

    # Call build_model_linreg() function
    model, J_storage = build_model_linreg(data["train_features"], data["train_target"])

    # Call the predict_linreg() method
    pred = predict_linreg(data["test_features"].to_numpy(), model["beta"], model["means"], model["stds"])

    # Change target test set to a numpy array
    target: np.ndarray = data["test_target"].to_numpy()

    # Calculate r2 score by calling a function
    r2: float = r2_score(target, pred)
    mse: float = mean_squared_error(target, pred)

    print(f"{feature}: R2: {r2} | MSE: {mse}")