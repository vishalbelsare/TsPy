from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 30, 10
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

from pprint import pprint

from base import RollingWindow

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.layers import Dense

def preprocess_data(series, sequence_length=10, normalise=True, pct_split=0.8):
    """Prepare data for LSTM network

    Parameters
    ----------
    series: pandas.Series
        Time series to be modeled
    sequence_length: int
        Size each sequence
    normalise: bool
        Flag for normalising each window
    
    Returns
    -------
    X_train: numpy.ndarray
        Training features matrix
    y_train: numpy.ndarray
        Training targets vector
    X_test: numpy.ndarray
        Testing features matrix
    y_test: numpy.ndarray
        Testing targets vector
    """
    def normaliser(series):
        """Normaliser for each (window) sub-series
        
        Parameters
        ----------
        series: pandas.Series
            Time series to be normalised
        
        Returns
        -------
        series: pandas.Series
            Normalised time series
        """
        return (series / series[0]) - 1
    
    rw = RollingWindow(sequence_length + 1)
    data = np.asarray( list( map( normaliser, rw.transform(series) ) ) ) if normalise else rw.transform(series)
    index_split = int( len(data) * pct_split )
    train = data[:index_split]
    test = data[index_split:]
    
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test

def build_model(layers, pct_dropout=0.2):
    """Build computational graph model
    
    Parameters
    ----------
    layers: list | [input, hidden_1, hidden_2, output]
        Dimensions of each layer
    pct_dropout: float | 0.0 to 1.0
        Percentage of dropout for hidden LSTM layers
    
    Returns
    -------
    model: keras.Model
        Compiled keras sequential model
    """
    if not isinstance(layers, list):
        raise TypeError('layers was expected to be of type %s, received %s' % (type([]), type(layers)))
    if len(layers) != 4:
        raise ValueError('4 layer dimentions required, received only %d' % len(layers))
    
    model = Sequential()
    
    model.add(LSTM(
        layers[1],
        input_shape=(layers[1], layers[0]),
        return_sequences=True,
        dropout=pct_dropout))        
    
    model.add(LSTM(
        layers[2],
        return_sequences=False,
        dropout=pct_dropout))
    
    model.add(Dense(
        layers[3],
        activation='linear'))
    
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict(model, array):
    """One step prediction
    
    Parameters
    ----------
    model: keras.Model
        Trained keras model
    array: numpy.ndarray
        Input features array
    
    Returns
    -------
    array: numpy.ndarray
        Predicted vector
    """
    _pred = model.predict(array)
    return np.reshape(_pred, (_pred.size,))

def backtest(model, series, sequence_length=10):
    """Backtest the model on the provided series
    
    Paramters
    ---------
    model: keras.Model
        Trained keras model
    series: pandas.Series
        Time series to be backtested
    sequence_length: int
        Size each sequence

    Returns
    -------
    backtest: base.BackTest
        Backtest object, implements methods: `summary`, `plot`, `outliers`
    """
    return

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, 'gx', label='Prediction')
        #plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed( 189 )
    window = 20
    N = 5000 + window

    #ts = pd.DataFrame.from_csv('./data/in/SPY.csv', header=0, index_col=0, parse_dates=True)['Adj Close']
    _ticker = 'AAPL.OQ'
    ts = pd.DataFrame.from_csv('./data/in/spx_vols_1.0_1M_2010-01-04_2017-04-10.csv', header=0, index_col=0, parse_dates=True)[_ticker].dropna()

    global_start_time = time.time()
    epochs  = 2
    seq_len = 50
    batch_size = 8
    pred_len = 1

    pct_split = 0.9

    print('> Loading data... ')

    X_train_pers, y_train_pers, X_test_pers, y_test_pers = preprocess_data(ts, seq_len, True, pct_split)
    X_train, y_train, X_test, y_test = X_train_pers[1:], y_train_pers[1:], X_test_pers[1:], y_test_pers[1:]
    X_train_pers, y_train_pers, X_test_pers, y_test_pers = X_train_pers[:-1], y_train_pers[:-1], X_test_pers[:-1], y_test_pers[:-1]

    print('> Data Loaded. Compiling...')

    lstm_model = build_model([1, seq_len, 100, 1], pct_dropout=0.5)

    lstm_model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1)

    predictions_test = predict_sequences_multiple(lstm_model, X_test, seq_len, pred_len)
    _predictions_test = np.asarray(predictions_test).ravel()

    predictions_train = predict_sequences_multiple(lstm_model, X_train, seq_len, pred_len)
    _predictions_train = np.asarray(predictions_train).ravel()

    print('Training duration (s) : ', time.time() - global_start_time)
    #plot_results_multiple(predictions, y_test, pred_len)

    plt.plot(_predictions_train)
    plt.plot(y_train)
    plt.plot(y_train_pers)
    plt.legend(['Model', 'Original', 'Baseline'])
    plt.title('Train Data')

    plt.figure()

    plt.plot(_predictions_test)
    plt.plot(y_test)
    plt.plot(y_test_pers)
    plt.legend(['Model', 'Original', 'Baseline'])
    plt.title('Test Data')


    mse_train = mean_squared_error(_predictions_train, y_train)
    mse_train_base = mean_squared_error(y_train_pers, y_train)

    print('MSE Train: %.3f,\n MSE Train Baseline: %.3f' % (mse_train, mse_train_base))

    mse_test = mean_squared_error(_predictions_test, y_test)
    mse_test_base = mean_squared_error(y_test_pers, y_test)

    print('MSE Test: %.3f,\n MSE Test Baseline: %.3f' % (mse_test, mse_test_base))

    plt.show()

    # out-of-sample prediction
    #plot_results_multiple(predictions, y_test, pred_len)

    # in-sample prediction -- anomaly detection specific
    #plot_results_multiple(predictions, y_train, pred_len)