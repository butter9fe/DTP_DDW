from library import get_features_targets, build_model_linreg, predict_linreg
from constants import FILE_NAME, CLEANED_FEATURES, TARGET
import numpy as np
import pandas as pd

class LungCancerPredictor:
    def __init__(self, EPI, AST, AQI, HDI, GDP, OADR, SR, OOP):
        features_array = [EPI, AST, AQI, HDI, GDP, OADR, SR, OOP]
        self.features = pd.DataFrame(np.array(features_array).reshape(-1, len(features_array)))
        self._build_model()

    def _build_model(self):
        df: pd.DataFrame = pd.read_csv(FILE_NAME)

        # Extract the features and the target
        df_features, df_target = get_features_targets(df, CLEANED_FEATURES, TARGET)

        # Split the data set into training and test
        # data = split_data(df_features, df_target)

        # Call build_model_linreg() function
        self.model, _ = build_model_linreg(df_features, df_target)

    def predict_lcr(self):
        pred = predict_linreg(self.features.to_numpy(), self.model["beta"], self.model["means"], self.model["stds"])
        return pred