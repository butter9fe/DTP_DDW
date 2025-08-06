from typing import Optional, Any    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns

#region Data Preparation
def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (1) Normalize the dataset -> between -1 to 1\n
    Formula: yhat = b0 + b1x

    Args:
        array (np.ndarray): numpy array to normalise (eg: training feature data)
        columns_means (Optional[np.ndarray], optional): _description_. Defaults to None.
        columns_stds (Optional[np.ndarray], optional): _description_. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: normalized array, column means, column stds
    """
    assert columns_means is None or columns_means.shape == (1, array.shape[1])
    assert columns_stds is None or columns_stds.shape == (1, array.shape[1])
    
    if columns_means is None: 
        columns_means = array.mean(axis=0).reshape(1, -1) # reshape output into 1 by N array shape 
    if columns_stds is None:
        print("hi")
        columns_stds = array.std(axis=0).reshape(1, -1)

    print(columns_stds)
    out: np.ndarray = (array - columns_means) / columns_stds
    
    assert out.shape == array.shape
    assert columns_means.shape == (1, array.shape[1])
    assert columns_stds.shape == (1, array.shape[1])
    return out, columns_means, columns_stds

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Convert to one-element lists if provided argument is not a list
    features = feature_names if isinstance(feature_names, list) else [feature_names]
    targets = target_names if isinstance(target_names, list) else [target_names]

    # Get dataframe of the provided features & targets
    df_feature: pd.DataFrame = df[features]
    df_target: pd.DataFrame = df[targets]
    return df_feature, df_target

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    # Get the number of rows
    m: int = np_feature.shape[0]
    
    # Create an array of 1s, with shape of m rows and 1 column
    ones_array = np.ones((m, 1))
    
    # Add column of constant 1s in first column
    X:np.ndarray = np.concatenate((ones_array, np_feature), axis = 1) # axis = 1 is to concatenate column wise
    return X

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_proportion: float=0.3, valid_proportion: float=0.3) -> dict[str, pd.DataFrame]:
    indexes: pd.Index = df_feature.index

    # Just to 'predict' randomness for autograder tests (if any) 
    if random_state != None:
        np.random.seed(random_state) 
    
    # Find indexes of the [Test] data set
    test_size: int = int(test_proportion * len(indexes))
    test_indexes = np.random.choice(indexes, test_size, replace=False)
    
    # find the indexes that are not selected by the test index
    removed_indexes = indexes.drop(test_indexes)

    # Find indexes of the [Validation] data set
    valid_size: int = int(valid_proportion * len(indexes))
    valid_indexes = np.random.choice(removed_indexes, valid_size, replace=False)

    # [Training] data set will be the remainining indexes
    train_indexes = removed_indexes.drop(valid_indexes)
    
    # time to create the dataframe of feature & target for each set (train & test)
    df_feature_train: pd.DataFrame = df_feature.loc[train_indexes, :]  
    df_feature_test: pd.DataFrame  = df_feature.loc[test_indexes, :]
    df_feature_valid: pd.DataFrame = df_feature.loc[valid_indexes, :]
    df_target_train: pd.DataFrame = df_target.loc[train_indexes, : ]
    df_target_test: pd.DataFrame = df_target.loc[test_indexes, :]
    df_target_valid: pd.DataFrame = df_target.loc[valid_indexes, :]
    
    return {"train_features": df_feature_train, "test_features": df_feature_test, "valid_features": df_feature_valid, "train_target": df_target_train, "test_target": df_target_test, "valid_target": df_target_valid}
#endregion Data Preparation

#region Linear Regression
def predict_linreg(array_feature: np.ndarray, beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    assert means is None or means.shape == (1, array_feature.shape[1])
    assert stds is None or stds.shape == (1, array_feature.shape[1])
    
    norm_data, _, _ = normalize_z(array_feature, means, stds) # (1) Standardize the feature using z-normalization
    X: np.ndarray = prepare_feature(norm_data) # (2) Change to Numpy array & add column of constant 1s
    result = calc_linreg(X, beta) # (3) Predict y values
    
    assert result.shape == (array_feature.shape[0], 1) # assert that the result vector is m by 1, where m is # data points
    return result

# yhat = b0hat + b1hat * x
def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Calculates linear regression\n
    Formula: yhat = b0hat + b1hat * x

    Returns:
        np.ndarray: result
    """
    result = np.matmul(X, beta)

    # we need to make sure that the shape of the result array tallies with X 
    # if we have N data points in the dataset, then the result array should have N x 1 dimension 
    assert result.shape == (X.shape[0], 1)
    return result

# J(B0, B1) = 1/2m * error_sq
def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    m: int = X.shape[0] # get the number of data in the dataset 
    predicted_y = calc_linreg(X, beta) # Linear Regression value -> yhat
    error = predicted_y - y  # this is error is a vector of shape m by 1 -> yhat - y
    
    # eg: error is [1 2 3] --> we want 1^2 + 2^2 + 3^2
    # we can do matmul: [[1 2 3]] ( 1 row 3 columns)  matmul  [1 2 3] (3 rows, 1 col) --> result is 1 row 1 col , e.g: [[14]]
    error_sq = np.matmul(error.T, error)
    J: np.ndarray = (1/(2*m)) * error_sq
    assert J.shape == (1,1) # 1 row 1 column 
    
    # we want to return scalar, so we need to take out the content of J
    return np.squeeze(J) # Same as J[0][0] -> returns axes in array with size 1, eg [[1]] -> [1]

def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray, 
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    # find size of data points
    m: int = X.shape[0]
    
    # create an array to store error value J at each iteration
    # it is an num_iters of gd x 1 vector
    J_storage: np.ndarray = np.zeros((num_iters, 1))
    for n in range(num_iters):
        # Eqn: beta1 = beta - (alpha) * (1/m) * (yhat - y) * X
        # (1) Compute derivative of error with this current beta
        yhat = calc_linreg(X, beta)
        # don't forget that matmul here "loops" through ALL m datapoints in the train set
        deriv: np.ndarray = np.matmul(X.T, (yhat - y)) # (yhat - y) * X
        
        # (2) Update the beta to be new beta
        beta = beta - alpha * (1/m) * deriv 
        
        # (3) Compute error value with this new beta
        J_storage[n] = compute_cost_linreg(X, y, beta)

    assert beta.shape == (X.shape[1], 1) # beta is a column vector 
    assert J_storage.shape == (num_iters, 1) 
    return beta, J_storage
#endregion Linear Regression

#region Evaluation Methods
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

def evaluate_model(model: dict[str, Any], valid_features: np.ndarray, valid_target: np.ndarray) -> dict[str, float]:
    """
    Evaluates model with validation dataset

    Returns:
        dict[str, float]: Dictionary with metrics containing:
            - cost: J score
            - r2
            - mse
    """
    # (1) Prepare X matrix with bias term
    valid_normalised, _, _ = normalize_z(valid_features, model["means"], model["stds"])
    x_valid = prepare_feature(valid_normalised)

    # (2) Calculate validation cost (J score!)
    validation_cost = compute_cost_linreg(x_valid, valid_target, model["beta"])

    # (3) Calculate other metrics
    valid_pred = predict_linreg(valid_features, model["beta"], model["means"], model["stds"])
    r2 = r2_score(valid_target, valid_pred)
    mse = mean_squared_error(valid_target, valid_pred)

    return {
        "cost": validation_cost,
        "r2": r2,
        "mse": mse
    }

#endregion Evaluation Methods

#region Model Building
def build_model_linreg(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    # Check if initial beta values are given
    if beta is None: 
        beta = np.zeros((df_feature_train.shape[1]+1, 1)) # Add one dimension to the feature_train array because of the b0 coefficient 
    assert beta.shape == (df_feature_train.shape[1]+1, 1) # To make sure if beta argument is given, then it conforms to the shape of the feature train

    # (1): Dataset Preparation
    # Normalize the features
    array_feature_train_z, means, stds = normalize_z(df_feature_train.to_numpy())

    # (2): Use Linear Regression
    # Prepare the X matrix and the target vector as ndarray 
    X: np.ndarray = prepare_feature(array_feature_train_z)
    target: np.ndarray = df_target_train.to_numpy()
    
    # (3) Perform gradient descent
    beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
    
    # (4) Store the output in model dictionary 
    model = {"beta": beta, "means":means, "stds": stds}

    # assert the shapes 
    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1) # make sure that beta vector is d by 1 
    assert model["means"].shape == (1, df_feature_train.shape[1]) # make sure that the means vector is also d-1 by 1 (1 per feature)
    assert model["stds"].shape == (1, df_feature_train.shape[1])  # make sure that the stds vector is also d-1 by 1 (1 per feature)
    assert J_storage.shape == (iterations, 1) # make sure we have recorded #iterations of error
    return model, J_storage

def build_model_with_validation(data: dict[str, pd.DataFrame],
                       feature_names: list[str],
                       alpha: float = 0.01,
                       iterations: int = 1500,
                       min_improve_threshold: float = 0.03) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """
    Trains the model with different features and use validation set to select the best ones

    Returns:
        tuple containing:
            - best_model (dictionary): Contains "beta", "means", "std" arrays
            - best_features (array): Best combination of features
    """
    best_model = {}
    best_features = []
    best_cost = np.inf
    all_combis = get_all_feature_combinations(feature_names)

    # Loop through every combination of features to build the best model!
    for features in all_combis:
       # (1) Extract the features for training & validation
       # Data was already split previously, so we're just taking a "subset" with the new feature combinations
        df_feature_train: pd.DataFrame = data["train_features"][features]
        df_feature_valid: pd.DataFrame = data["valid_features"][features]

        # (2) Splitting was already done in previous stage

        # (3) Build model
        model, _ = build_model_linreg(df_feature_train, data["train_target"], alpha=alpha, iterations=iterations)
        
        # (4) Evaluate model
        evaluation = evaluate_model(model, df_feature_valid, data["valid_target"])

        print(f"Model: {features} | {evaluation}")

        # To prevent overfitting validation data & adding unnecessary complexity,
        # we only add additional complexity from the previous features if the improvement is over the desired amount
        # previously just: evaluation["cost"] < best_cost
        if ((evaluation["cost"] < best_cost and len(features) <= len(best_features)) \
            or (evaluation["cost"] < best_cost * (1 - min_improve_threshold))):
            print("Better model found!")
            best_cost = evaluation["cost"]
            best_features = features
            best_model = model

    return best_model, best_features
#endregion Model Building

#region Helper Methods
def combinations(iterable, r):
    """
    Source: https://docs.python.org/3/library/itertools.html#itertools.combinations
    Return r length subsequences of elements from the input iterable. 
    
    Examples:
    combinations('ABCD', 2) → AB AC AD BC BD CD
    combinations(range(4), 3) → 012 013 023 123
    """
    pool = list(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))

    yield list(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield list(pool[i] for i in indices)

def get_all_feature_combinations(features: pd.DataFrame) -> list[str]:
    """
    Get all possible combinations of feature names
    """
    all_combis = []
    for i in range(1, len(features) + 1):
        for combo in list(combinations(features, i)):
            all_combis.append(combo)

    return all_combis
#endregion