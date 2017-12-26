from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import tensorflow as tf

from tspy.model import _Model

import collections


class DNNState:
    """Deep Neural Network State"""

    def __init__(self, window=0, history=[]):
        if not isinstance(window, int):
            raise TypeError('type(window)=%s!=int' % type(window))
        self.window = window
        if not isinstance(history, collections.Iterable):
            raise TypeError(
                'type(history)=%s!=collections.Iterable' % type(history))
        self.history = history


class DNN(_Model):
    """Deep Neural Network
    based on `tf.estimator.DNNRegressor`
    """

    def __init__(self, state=DNNState(), name='DNN'):
        self.state = DNNState()
        self.name = name

    def fit(self, X, y, num_epochs=100):
        return self

    def predict(self, X):
        return []

    def score(self, X, y):
        return 0

    def next(self):
        raise NotImplementedError
