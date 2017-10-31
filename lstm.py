from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import LSTM as _LSTM
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.layers import Dense

class LSTM(object):
    """Long Short Term Memory Regressor Class"""

    def __init__(self, layers, pct_dropout=0.2):
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
        
        self.model = Sequential()
        
        self.model.add(_LSTM(
            layers[1],
            input_shape=(layers[1], layers[0]),
            return_sequences=True,
            dropout=pct_dropout))        
        
        self.model.add(_LSTM(
            layers[2],
            return_sequences=False,
            dropout=pct_dropout))
        
        self.model.add(Dense(
            layers[3],
            activation='linear'))

        self.model.compile(loss="mse", optimizer="rmsprop")

    def fit(self, X, y, **kwargs):
        """Train the model"""
        self.model.fit(X, y, **kwargs)
    
    def predict(self, series):
        """Prediction using provided series"""
        return self.model.predict(series)