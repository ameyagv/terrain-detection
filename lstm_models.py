"""
    Script to define LSTM models
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.optimizers import Adam

HIDDEN_SIZE = 128


# LSTM_1 - LSTM - A simple LSTM loop with configurations of dropout and learning rate
def LSTM_1(X, y):
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
    n_inputs, n_features, n_outputs = 30, 6, 4
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(n_inputs, n_features)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
    return model


# LSTM_2 - BiDirectional LSTM followed by LSTM
def LSTM_2(X, y):
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
        - Learning Rate: 0.001
    :Returns
        model
    """
    n_inputs, n_features, n_outputs = 30, 6, 4
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE), input_shape=(n_inputs, n_features)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
    return model


 # LSTM 3 - BiDirectional LSTM followed by LSTM + LSTM
def LSTM_3(X, y):
    """
    :Args
        X, y: np.array(), np.array()
    :LSTM 3
        - BiDirectional LSTM Layer
        - LSTM Layer
        - Hidden Units: 128
        - LSTM Layer
        - Hidden Units: 128
        - 2 Dense layers:
            Dense1: 128 + ReLU
            Dense2: 4 + SoftMax
        - Dropout Rate: 0.3, 0.2
    :Returns
        model
    """
    n_inputs, n_features, n_outputs = 30, 6, 4
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), input_shape=(n_inputs, n_features)))
    model.add(Dropout(rate=0.3))
    model.add(LSTM(HIDDEN_SIZE, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(HIDDEN_SIZE))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=2)
    return model


# Layer of LSTM followed by LSTM
def LSTM_4(X, y):
    """
    :Args
        X, y: np.array(), np.array()
    :LSTM 4
        - LSTM Layer
        - Hidden Units: 128
        - LSTM Layer
        - Hidden Units: 128
        - 2 Dense layers:
            Dense1: 128 + ReLU
            Dense2: 4 + SoftMax
        - Dropout Rate: 0.5
    :Returns
        model
    """
    n_inputs, n_features, n_outputs = 30, 6, 4
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(rate=0.5))
    model.add(LSTM(HIDDEN_SIZE))
    model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
    return model


# Helper Function to Save the Model for future use
def save_model(model):
    model_yaml = model.to_yaml()
    with open('model/model.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights('model/model.h5')
    print('Model Saved Successfully :thumbs_up:')
