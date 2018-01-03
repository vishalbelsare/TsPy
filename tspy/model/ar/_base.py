from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import tensorflow as tf

from tspy.model._base import _Model


class _ARState:
    """AR Model State."""

    def __init__(self, window):
        """Constructs a `_ARState` instance.

        Parameters
        ----------
        window: int
            Window size
        """
        if not isinstance(window, int):
            raise TypeError('type(window)=%s!=int' % type(window))
        self.window = window
        self._history = np.array([float()] * window)

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, array):
        if not isinstance(array, np.ndarray):
            array = np.array(array).ravel()
        if len(array) != self.window:
            raise ValueError('%d=len(history)!=self.window=%d' %
                             (len(array), self.window))
        self._history = array.reshape(1, self.window)


class _ARModel(_Model):
    """AR models base class.

    API
    ---
    _fit: self
        Fit model method
    _predict: numpy.ndarray
        Predict method
    _score: float
        Model evaluation method
    """

    def __init__(self, window, name='AR'):
        """Constructs a `_ARModel` instance.

        Parameters
        ----------
        window: int
            Window size
        name: str
            Model name
        """
        super(_ARModel, self).__init__(name)
        self.state = _ARState(window)

    def _after_fit(self, X, y):
        """Method called after `fit`.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector
        """
        self._set_state(X, y)

    def _after_predict(self, X, y_hat):
        """Method called after `predict`.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y_hat: numpy.ndarray | NoneType
            Target vector
        """
        self._set_state(X, y_hat)

    def _set_state(self, X, y):
        """Update `self.state.history`.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector
        """
        self.state.history = np.append(X[-1, :-1], y[-1])

    def next(self, steps=1):
        _next = []
        for i in range(steps):
            _next.append(self.predict(self.state.history))
        return np.array(_next).reshape(steps, 1)
