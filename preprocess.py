import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalize_data(input_data, data_columns):
    """
        Function to Scale the data
    """
    standard_scaler = StandardScaler()
    standard_scaler = standard_scaler.fit(input_data[data_columns])
    input_data.loc[:, data_columns] = standard_scaler.transform(input_data[data_columns].to_numpy())
    return input_data


def encode_labels(y):
    """
       Helper function for OneHotEncoding of Labels
    """
    encoder = OneHotEncoder(sparse=False)
    encoder = encoder.fit(y)
    y_encoded = encoder.transform(y)
    return y_encoded


def match_frequency(X,y):
    """
        Upsampling of Labels
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


def get_windowed_data(X, y, n_steps, step_size):
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


def get_ts_format(X_list, y_list, n_steps, step_size):
    """
        Get the Time Series Data to feed the model
    """
    list_X, list_y = [], []
    for each in range(len(y_list)):
        X, y = match_frequency(X_list[each], y_list[each])
        # X_values =
        X = normalize_data(X, list(X.columns.values))
        y, X = get_windowed_data(X, y, n_steps, step_size)
        list_X.append(X)
        list_y.append(y)
    merged_X = np.concatenate(list_X)
    merged_y = np.concatenate(list_y)

    return merged_X, merged_y
