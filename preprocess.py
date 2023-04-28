"""
    Python Script for Data PreProcessing and PostProcessing
"""
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Helper function to Normalize the Data
def normalize_data(input_data, data_columns):
    """
        Args:
            input_data -> np.array
            data_columns -> np.array
        Returns:
            input_data: Normalized Input Data -> np.array
    """
    standard_scaler = StandardScaler()
    standard_scaler = standard_scaler.fit(input_data[data_columns])
    input_data.loc[:, data_columns] = standard_scaler.transform(input_data[data_columns].to_numpy())
    return input_data


# Helper function to OneHotEncode the Labels
def encode_labels(y):
    """
       Args:
            y: Labels -> np.array
       Returns:
            y: encoded labels -> np.array
    """
    encoder = OneHotEncoder(sparse=False)
    encoder = encoder.fit(y)
    y_encoded = encoder.transform(y)
    return y_encoded


# Helper function to upsamle the labels to match frequency of X and y
def match_frequency(X,y):
    """
        Args:
            X: Input dataframe -> pd.DataFrame
            y: target_labels -> pd.DataFrame
        Returns:
            X_df: Input Dataframe -> pd.DataFrame
            upsampled_targets: upsampled y -> List()
    """
    X_df = pd.read_csv(X)
    y_df = pd.read_csv(y)

    upsampled_y = []
    for i in y_df.iterrows():
        upsampled_y += [i[1][0]] * 4

    upsampled_targets = pd.DataFrame(upsampled_y)
    difference = X_df.shape[0] - upsampled_targets.shape[0]
    X_df = X_df.iloc[:-difference, :]

    return X_df, upsampled_targets


# Function to get Windowed Data to feed to LSTM model
def get_windowed_data(X, y, n_steps, step_size):
    """
    Args:
        X: input dataset -> List()
        y: label dataset -> List()
        n_steps: Number of Steps -> int
        step_size: Size of steps -> int
    Returns:
        windowed_y: Windowed targets -> List()
        windowed_x: Windowed inputs -> List()
    """
    X_values, y_values = [], []
    steps = len(X) - n_steps
    for i in range(0, steps, step_size):
        inputs = X.iloc[i:(i + n_steps)].values
        targets = y.iloc[i:(i + n_steps)]
        common_label = stats.mode(targets)[0][0]
        X_values.append(inputs)
        y_values.append(common_label)
    windowed_y = np.array(y_values).reshape(-1, 1)
    windowed_x = np.array(X_values)

    return windowed_y, windowed_x


# Function to Get the Time Series Format to feed data to LSTM
def get_ts_format(X_list, y_list, n_steps, step_size):
    """
        Args:
            x_list: Windowed X -> List()
            y_list: Normalized Encoded Windowed y -> List()
            n_steps: Number of Steps -> int
            step_size: Size of steps -> int
        Returns:
            merged_x: TS format X -> np.array
            merged_y: TS format y -> np.array
    """
    list_X, list_y = [], []
    for each in range(len(y_list)):
        X, y = match_frequency(X_list[each], y_list[each])
        X = normalize_data(X, list(X.columns.values))
        y, X = get_windowed_data(X, y, n_steps, step_size)
        list_X.append(X)
        list_y.append(y)
    merged_X = np.concatenate(list_X)
    merged_y = np.concatenate(list_y)

    return merged_X, merged_y


# Function to get the most repeated value of labels after prediction
def calculate_mode(y):
    """
        Args:
            output: predicted labels -> np.array
        Returns:
            output_actual -> np.array
    """
    output_downsampled = []
    for i in range(0, y.shape[0], 4):
        y_list = list(y[i:i + 4])
        mode, _  = stats.mode(y_list)
        output_downsampled.append(mode)
    return np.array(output_downsampled)

# Function to create Test Dataset in TimeSeries Format
def get_ts_testdata(X, n_steps, step_increment):
    """
        Args:
            X -> np.array
            n_steps -> int
            step_increment -> int
        Returns:
            X_values -> np.array()
    """
    X_list = []
    for i in range(0, len(X) - n_steps, step_increment):
        extract = X.iloc[i:(i + n_steps)].values
        X_list.append(extract)
    return np.array(X_list)
