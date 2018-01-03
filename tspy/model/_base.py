from __future__ import absolute_import

# scientific computing
import numpy as np
import tensorflow as tf
import pandas as pd


class _Model:
    """Base model class.

    API
    ---
    _fit: self
        Fit model method
    _after_fit: NoneType <optional>
        Method called after `fit`
    _predict: numpy.ndarray
        Predict method
    _after_predict: NoneType <optional>
        Method called after `predict`
    _score: float
        Model evaluation method
    next: numpy.ndarray
        Predictions based on `state`
    """

    def __init__(self, name=None):
        """Constructs a `_Model` instance.

        Parameters
        ----------
        name: str
            Model name
        """
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
        model: self
            Fitted model
        """
        self._fit(X, y, **kwargs)
        self._fitted = True
        try:
            self._after_fit(X, y)
        except NotImplementedError:
            print('%s: <optional> `_after_fit` not implemented' % self)
        return self

    def _fit(self, X, y, **kwargs):
        raise NotImplementedError

    def _after_fit(self, X, y, **kwargs):
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
            try:
                self._after_predict(X, y_hat)
            except NotImplementedError:
                print('%s: <optional> `_after_predict` not implemented' % self)
            return y_hat

    def _predict(self, X):
        raise NotImplementedError

    def _after_predict(self, X, y_hat):
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
