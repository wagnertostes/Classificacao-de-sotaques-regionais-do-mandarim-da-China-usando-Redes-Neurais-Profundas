
from numpy.lib.shape_base import expand_dims
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.ops.gen_array_ops import ExpandDims, Reshape


def get_model(model, input_shape, num_labels):
    if model == 'cnn1d':
        return traditional_cnn_1d_model(input_shape, num_labels)
    elif model == 'cnn2d':
        return traditional_cnn_2d_model(input_shape, num_labels)
    elif model == 'cnn2d_2':
        return traditional_cnn_2d_model_2(input_shape, num_labels)
    elif model == 'wagner_lstm':
        return wagner_lstm(input_shape, num_labels)
    elif model == 'wagner_bilstm':
        return wagner_bilstm(input_shape, num_labels)
    elif model == 'wagner_lstm_conv1d_1':
        return wagner_lstm_conv1d_1(input_shape, num_labels)
    elif model == 'wagner_bilstm_conv1d_1':
        return wagner_bilstm_conv1d_1(input_shape, num_labels)
    elif model == 'wagner_lstm_conv1d_2':
        return wagner_lstm_conv1d_2(input_shape, num_labels)
    elif model == 'wagner_bilstm_conv1d_2':
        return wagner_bilstm_conv1d_2(input_shape, num_labels)
    elif model == 'wagner_lstm_conv1d_3':
        return wagner_lstm_conv1d_3(input_shape, num_labels)
    elif model == 'wagner_bilstm_conv1d_3':
        return wagner_bilstm_conv1d_3(input_shape, num_labels)
    else:
        msg = f"Model {model} not found."
        raise Exception(msg)


def traditional_cnn_1d_model(input_shape, num_labels):
    print("input shape:", input_shape)
    model = models.Sequential([
        layers.Input(shape=(input_shape[0], 1), name='audio'),
        layers.Reshape((input_shape[0], 1)), 
        layers.Conv1D(filters=64, kernel_size=5, strides=5, activation='relu', name='c_1'),
        layers.MaxPooling1D(pool_size=5, name='maxpool_1'),
        layers.Dropout(0.4),
        layers.Conv1D(filters=64, kernel_size=5, strides=5, activation='relu', name='c_2'),
        layers.MaxPooling1D(pool_size=5, name='maxpool_2'),
        #layers.Dropout(0.2),
        layers.Conv1D(filters=64, kernel_size=5, strides=5, activation='relu', name='c_3'),
        layers.GlobalMaxPooling1D(name='globalpool_1'),
        layers.Dense(128, activation='relu', name='dense_1'),
        #layers.Dropout(0.3),
        #layers.Dense(128, activation='relu', name='dense_2'),
        #layers.Dropout(0.4),
        layers.Dense(num_labels, name='output'),
    ])
    return model


def traditional_cnn_2d_model(input_shape, num_labels):
    kernel_size = (3, 5)
    strides = (2, 3)
    pool_size_1 = (2, 4)
    pool_size_2 = (2, 2)
    
    model = models.Sequential([
        layers.Input(shape=(input_shape[0], input_shape[1], 1), name='spectrogram'),
        layers.Reshape((input_shape[0], input_shape[1], 1), name='reshape'),
        layers.Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation='relu', name='c_1'),
        layers.MaxPooling2D(pool_size=pool_size_1, name='maxpool_1'),
        layers.Dropout(0.4),
        layers.Conv2D(filters=64, kernel_size=kernel_size, strides=strides, activation='relu', name='c_2'),
        layers.MaxPooling2D(pool_size=pool_size_2, name='maxpool_2'),
        #layers.Dropout(0.2),
        layers.Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation='relu', name='c_3'),
        #layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_1'),
        #layers.Dropout(0.2),
        layers.Dense(num_labels, name='output'),
    ])
    return model


def traditional_cnn_2d_model_2(input_shape, num_labels):
    model = models.Sequential([
        layers.Input(shape=(input_shape[0], input_shape[1]), name='spectrogram'),
        layers.Reshape((input_shape[0], input_shape[1], 1), name='reshape'),
        layers.Conv2D(filters=32, kernel_size=(1, input_shape[1]), strides=(1, input_shape[1]), activation='elu', name='c_1'),
        layers.MaxPooling2D(pool_size=(4, 1), name='maxpool_1'),
        layers.Dropout(0.4),
        layers.Conv2D(filters=32, kernel_size=(5, 1), strides=(2, 1), activation='elu', name='c_2'),
        layers.MaxPooling2D(pool_size=(3, 1), name='maxpool_2'),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dropout(0.2),
        layers.Dense(num_labels, name='output'),
    ])
    return model


def wagner_lstm(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Dense(32, activation='sigmoid'),
        layers.Dropout(0.6),
        layers.LSTM(64, use_bias=False, implementation=1),
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dense(128, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model


def wagner_bilstm(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Dense(32, activation='sigmoid'),
        layers.Dropout(0.6),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dense(128, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model


def wagner_lstm_conv1d_1(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Reshape((1, input_shape[0], input_shape[1])),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Dropout(0.6),
        layers.Reshape((input_shape[0], 30)),
        layers.LSTM(64),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1000, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model


def wagner_bilstm_conv1d_1(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Reshape((1, input_shape[0], input_shape[1])),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Dropout(0.6),
        layers.Reshape((input_shape[0], 30)),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1000, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model


def wagner_lstm_conv1d_2(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Reshape((1, input_shape[0], input_shape[1])),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Dense(30, activation='sigmoid'),
        layers.Dropout(0.6),
        layers.Reshape((input_shape[0], 30)),
        layers.LSTM(64),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(100, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model


def wagner_bilstm_conv1d_2(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Reshape((1, input_shape[0], input_shape[1])),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Dense(30, activation='sigmoid'),
        layers.Dropout(0.6),
        layers.Reshape((input_shape[0], 30)),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(100, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model

def wagner_lstm_conv1d_3(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Reshape((1, input_shape[0], input_shape[1])),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Dropout(0.6),
        layers.Reshape((input_shape[0], 30)),
        layers.LSTM(64),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(100, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model

def wagner_bilstm_conv1d_3(input_shape, num_labels):
    model = models.Sequential([
        layers.Input((input_shape[0], input_shape[1])),
        layers.Reshape((1, input_shape[0], input_shape[1])),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Conv1D(filters=30, kernel_size=1, activation='relu'),
        layers.Dropout(0.6),
        layers.Reshape((input_shape[0], 30)),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(100, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(num_labels),
    ])
    return model
