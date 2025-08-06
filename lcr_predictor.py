from library import predict_linreg
from web_library import get_variable_median
from constants import CLEANED_FEATURES, BETA, MEANS, STDS
import pandas as pd

class LungCancerPredictor:
    def __init__(self, inputs: dict[str, float]):
        features = {}
        for key, value in inputs.items():
            features[key] = value if value != None else get_variable_median(key)

        self.features = pd.Series(features).to_frame().T # Convert dictionary to series as a row
        self.features = self.features[CLEANED_FEATURES] # Reorder features to match training data

    def predict_lcr(self):
        pred = predict_linreg(self.features.to_numpy(), BETA, MEANS, STDS)
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