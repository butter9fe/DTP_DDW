from library import predict_linreg
from web_library import get_variable_median, get_or_from_ancestry
from constants import CLEANED_FEATURES, BETA, MEANS, STDS
import pandas as pd
import numpy as np

class LungCancerPredictor:
    def __init__(self, inputs: dict[str, float]):
        features = {}
        for key, value in inputs.items():
            if (key != "ANC"):
                features[key] = value if value != None else get_variable_median(key)

        self.features = pd.Series(features).to_frame().T # Convert dictionary to series as a row
        self.features = self.features[CLEANED_FEATURES] # Reorder features to match training data
        self.or_value = get_or_from_ancestry(inputs["ANC"])

    def predict_lcr(self):
        # Remember this gives us LCR_OR!
        pred = predict_linreg(self.features.to_numpy(), BETA, MEANS, STDS)

        # Need to multiply back to get raw LCR value
        return  np.squeeze(pred) * self.or_value