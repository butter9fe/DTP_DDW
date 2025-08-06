from library import get_features_targets, build_model_linreg, predict_linreg, split_data
from web_library import get_variable_median
from constants import FILE_NAME, CLEANED_FEATURES, TARGET, BETA
import pandas as pd

class LungCancerPredictor:
    def __init__(self, inputs: dict[str, float]):
        features = {}
        for key, value in inputs.items():
            features[key] = value if value != None else get_variable_median(key)

        self.features = pd.Series(features).to_frame().T # Convert dictionary to series as a row
        self.features = self.features[CLEANED_FEATURES] # Reorder features to match training data
        self._build_model()

    def _build_model(self):
        df: pd.DataFrame = pd.read_csv(FILE_NAME)

        # Extract the features and the target
        df_features, df_target = get_features_targets(df, CLEANED_FEATURES, TARGET)

        # Split the data set into training and test
        #data = split_data(df_features, df_target)

        # Call build_model_linreg() function
        self.model, _ = build_model_linreg(df_features, df_target)

    def predict_lcr(self):
        pred = predict_linreg(self.features.to_numpy(), self.model["beta"], self.model["means"], self.model["stds"])
        return pred

# test = LungCancerPredictor({
#     "EPI": 19.4,
#     "AST": 25.04,
#     "AQI": 28.2,
#     "HDI": 0.609,
#     "GDP": 5177.4326,
#     "OADR": 10.05489,
#     "SR": 44.4,
#     "OOP": 78.24772727,
# })
# print(test.predict_lcr())