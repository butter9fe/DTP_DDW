# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from typing import Optional, Any
from constants import FILE_NAME, OR_FILE_NAME, ALL_FEATURES, TARGET

# Loading data
df: pd.DataFrame = pd.read_csv(FILE_NAME)  # Main Dataset
or_df: pd.DataFrame = pd.read_csv(OR_FILE_NAME)  # OR Data

# Before any analysis/cleaning, let's separate our features and target!
def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get features and targets from dataframe
    """
    # Convert to one-element lists if provided argument is not a list
    features = feature_names if isinstance(feature_names, list) else [feature_names]
    targets = target_names if isinstance(target_names, list) else [target_names]

    # Get dataframe of the provided features & targets
    df_feature: pd.DataFrame = df[features]
    df_target: pd.DataFrame = df[targets]
    return df_feature, df_target

# ALL_FEATURES: EPI, AST, AQI, HDI, GDP, PWD_A, OADR, SR, OOP, LCR, MAJ, LCR_RAF, PDPC, CO2, HUM
# TARGET: LCR_OR
df_feature, df_target = get_features_targets(df, ALL_FEATURES, TARGET)
display(df_feature)
display(df_target)

"""
Clean & Analyze your data
Use python code to:

Clean your data
Calculate Descriptive Statistics and other statistical analysis
Visualization with meaningful analysis description
"""

# Clean Data
# Note that while collecting the data in the .csv, we have already dropped rows that contained too many missing values,
# as well as other cleaning steps such as removing duplicates, etc.

def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize the dataset -> between -1 to 1
    Formula: yhat = b0 + b1x

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: normalized array, column means, column stds
    """
    assert columns_means is None or columns_means.shape == (1, array.shape[1])
    assert columns_stds is None or columns_stds.shape == (1, array.shape[1])
    
    if columns_means is None: 
        columns_means = array.mean(axis=0).reshape(1, -1) # reshape output into 1 by N array shape 
    if columns_stds is None:
        columns_stds = array.std(axis=0).reshape(1, -1)

    out: np.ndarray = (array - columns_means) / columns_stds
    
    assert out.shape == array.shape
    assert columns_means.shape == (1, array.shape[1])
    assert columns_stds.shape == (1, array.shape[1])
    return out, columns_means, columns_stds

array_features_normalized, means, stds = normalize_z(df_feature.to_numpy())

# descriptive
def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    ymean: np.ndarray = np.mean(y)
    diff: np.ndarray = y - ymean #(y - ybar)
    sstot: np.ndarray = np.matmul(diff.T, diff) # (y - ybar)^2
    error: np.ndarray = y - ypred # (y - yhat)
    ssres: np.ndarray = np.matmul(error.T, error) # (y - yhat)^2
    return 1 - np.squeeze(ssres/sstot) # remember to squeeze the value out of the matrix form because r^2 is a scalar, not a 1-element vector [[r^2]] 

def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    n: int = target.shape[0] # number of data points 
    error = target - pred #(y - yhat)^2
    error_sq = np.matmul(error.T, error)
    return 1/n * np.squeeze(error_sq)

# visualization
sns.set()
for index, feature in enumerate(ALL_FEATURES):
    feature_row = df_feature[feature]
    slope, intercept = np.polyfit(feature_row, df_target, 1)
    line_y = slope * feature_row + intercept

    plt.scatter(feature_row, df_target, label=feature)
    plt.plot(feature_row, line_y, color='blue', label='Best Fit Line')
    plt.title(f"{feature} vs LCR_OR")
    plt.show()