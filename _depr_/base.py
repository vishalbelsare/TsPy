import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class BackTest(object):
    """Backtest result class"""

    def __init__(self, series, series_hat):
        self.series = series
        self.index = self.series.index
        self.prediction = pd.Series(series_hat, index=self.index, name='Prediction')
        _resid = self.prediction - self.series
        self.zscore = (_resid - _resid.mean()) / _resid.std()
        self.zscore.name = 'zscore'
        self.mse = mean_squared_error(self.prediction, self.series)
        _persist = series[1:]
        self.mse_baseline = mean_squared_error(_persist, self.series[:-1])

    def summary(self):
        """Metrics summary of the backtest

        Returns
        -------
        df: pandas.DataFrame
            Table with:
            * Mean Squared Error of predictor (MSE)
            * Mean Squared Error of Persistance model (MSE Baseline)
        """
        return pd.DataFrame({'MSE': self.mse, 'MSE Baseline': self.mse_baseline}, index=['Metrics'])


    def outliers(self, threshold=1.96):
        _mask = np.abs(self.zscore) > threshold
        return self.index[_mask]
    

    def plot(self, **kwargs):
        self.series.plot()
        self.prediction.plot()
        plt.legend(['Original Series', 'Model', ])
        if kwargs.get('title'):
            plt.title(kwargs['title'])
        _threshold = kwargs.get('threshold', 1.96)
        _mask = np.abs(self.zscore) > _threshold
        _ymax, _ymin = self.prediction.max(), self.prediction.min()
        if kwargs.get('anomalies', True):
            plt.fill_between( self.index, _ymin, _ymax, where=_mask, facecolor='#c4ad66', alpha=0.35 )
        if kwargs.get('zscore', False):
            plt.figure()
            self.zscore.plot()
        


class _BaseTrasnformer(sklearn.base.TransformerMixin):
    """`sklearn` transform interface, base class"""

    def fit(self, X, y=None, **fit_params):
        """enable cascading by default"""
        return self
    
    def transform(self, seq, **transform_params):
        """Transform function
        
        Parameters
        ----------
        seq: n-dimensional sequential data structure (i.e list, pandas.Series, numpy.ndarray)
        
        Returns
        -------
        container: n-dimensional container data structure
        """
        raise NotImplementedError


class RollingWindow(_BaseTrasnformer):
    """Optimized rolling window transformer"""

    def __init__(self, window, batch=False, normaliser=None):
        """Constructor
        
        Parameters
        ----------
        window: int
            Window size
        batch: bool
            Format data for batch training -- Deep Learning friendly
            LSTM: [samples, time steps, features]
        normaliser: lambda numpy.ndarray -> numpy.ndarray
            Normaliser for each (window) sub-series
        """
        self.window = window
        self.batch = batch
        self.normaliser = normaliser
    
    def transform(self, seq, **transform_params):
        """Rolling window transform"""
        _shape = seq.shape[:-1] + (seq.shape[-1] - self.window + 1, self.window)
        _strides = seq.strides + (seq.strides[-1],)
        _batches = np.lib.stride_tricks.as_strided(seq, shape=_shape, strides=_strides)[:-1]
        _array = np.asarray( list( map( self.normaliser, _batches ) ) ) if self.normaliser else _batches
        return _array.reshape(_array.shape[0], _array.shape[1], 1) if self.batch else _array
