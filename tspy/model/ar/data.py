from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import tensorflow as tf
import pandas as pd

import tspy


def AR_Data(ticker, window, batch_size=1):
    """Autoregressive data generator.

    Parameters
    ----------
    ticker: str
        Ticker name
    window: int
        Window size
    batch_size: int
        Number of elements per batch

    Returns
    -------
    dataset: tf.data.Dataset
        Tensorflow Dataset
    """
    series = tspy.adapters.Finance.Prices([ticker])[ticker]
    X, y = Xy(series, window)
    _dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = _dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return dataset, iterator


def rolling(series, window):
    """Rolling window.

    Parameters
    ----------
    series: list | numpy.ndarray | pandas.Series
        Sequential data
    window: int
        Window size

    Returns
    -------
    matrix: numpy.ndarray
        Matrix of rolling windowed series
    """
    if isinstance(series, list):
        series = np.array(series)
    shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
    strides = series.strides + (series.strides[-1],)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)


def Xy(series, window):
    """Supervised data format

    Parameters
    ----------
    series: list | numpy.ndarray | pandas.Series
        Sequential data
    window: int
        Window size

    Returns
    -------
    X: numpy.ndarray
        Features matrix
    y: numpy.ndarray
        Target vector
    """
    raw = rolling(series, window + 1)
    X = raw[:, :-1]
    y = raw[:, -1]
    return X, y
