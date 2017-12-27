from __future__ import absolute_import

# scientific computing
import numpy as np
import tensorflow as tf
import pandas as pd


class _Model:

    def __init__(self, name=None):
        self.name = name
        self._fitted = False

    def fit(self, X, y, **kwargs):
        """Model fitting.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector

        Returns
        -------
        model: DNN
            Fitted model
        """
        self._fit(X, y, **kwargs)
        self._fitted = True
        if callable(getattr(self, '_set_state', None)):
            self._set_state(X, y, **kwargs)
        return self

    def _fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        """Predict method.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix

        Returns
        -------
        y_hat: numpy.ndarray | NoneType
            Predicted target vector

        Raises
        ------
        exception: AssertionError
            `predict` before `fit` call
        """
        if not self._fitted:
            raise AssertionError('`predict` called on an unfitted model')
        else:
            y_hat = self._predict(X)
            if callable(getattr(self, '_set_state', None)):
                self._set_state(X, y_hat)
            return y_hat

    def _predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        """Evaluate model.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            True target vector

        Returns
        -------
        loss: float
            Mean squared error loss
        """
        return self._score(X, y)

    def _score(self, X, y):
        raise NotImplementedError

    def __call__(self, X, y=None):
        """Callable object for `predict` / `score` alias.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            True target vector

        Returns
        -------
        y_hat: numpy.ndarray | NoneType
            Predicted target vector

        Raises
        ------
        exception: AssertionError
            `predict` before `fit` call
        """
        if y is None:
            return self.predict(X)
        else:
            return self.score(X, y)

    def next(self, steps=1):
        raise NotImplementedError
