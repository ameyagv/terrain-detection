import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.optimizers import Adam


HIDDEN_SIZE = 128


# LSTM_1 - LSTM
def LSTM_1(X,y):
    """
    :Args
        X, y: np.array(), np.array()
    :LSTM 1
        - LSTM Layer
        - Hidden Units: 128
        - 2 Dense layers:
            Dense1: 128 + ReLU
            Dense2: 4 + SoftMax
        - Dropout Rate: 0.5
        - Learning Rate: 0.001
    :Returns
        model
    """
    n_inputs, n_features, n_outputs = X.shape[0], X.shape[1], y.shape[1]
    #n_inputs, n_features, n_outputs = 40, 6, 4
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE,input_shape=(n_inputs, n_features)))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(HIDDEN_SIZE, activation = 'relu'))
    model.add(Dense(n_outputs, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer =Adam(learning_rate=0.001), metrics = ['acc'])
    return model


# LSTM_2 - BiDirectional LSTM
def LSTM_2(X,y):
    """
    :Args
        X, y: np.array(), np.array()
    :LSTM 2 -
        - BiDirectional LSTM Layer
        - Hidden Units: 128
        2 Dense layers:
            Dense1: 128 + ReLU
            Dense2: 4 + SoftMax
        - Dropout Rate: 0.5
        - Learning Rate: 0.01
    :Returns
        model
    """
    n_inputs, n_features, n_outputs = X.shape[0], X.shape[1], y.shape[1]
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE,input_shape=(n_inputs, n_features))))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(HIDDEN_SIZE, activation = 'relu'))
    model.add(Dense(n_outputs, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.01), metrics = ['acc'])
    return model


# LSTM_3 - BiDirectional LSTM + LSTM
def LSTM_3(X,y):
    """
    :Args
        X, y: np.array(), np.array()
    :LSTM 1
        - BiDirectional LSTM Layer
        - LSTM Layer + ReLU
        - Hidden Units: 128
        - 2 Dense layers:
            Dense1: 128 + ReLU
            Dense2: 4 + SoftMax
        - Dropout Rate: 0.2, 0.1
        - Learning Rate: 0.001
    :Returns
        model
    """
    n_inputs, n_features, n_outputs = X.shape[0], X.shape[1], y.shape[1]
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE,input_shape=(n_inputs, n_features))))
    model.add(Dropout(rate = 0.2))
    model.add(LSTM(HIDDEN_SIZE,activation='relu'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(HIDDEN_SIZE, activation = 'relu'))
    model.add(Dense(n_outputs, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer =Adam(learning_rate=0.001), metrics = ['acc'])
    return model
