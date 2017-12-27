from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import tensorflow as tf

from tspy.model import _Model


class DNNState:
    """Deep Neural Network Model State."""

    def __init__(self, window):
        """Constructs a `DNNState` instance.

        Parameters
        ----------
        window: int
            Window size
        """
        if not isinstance(window, int):
            raise TypeError('type(window)=%s!=int' % type(window))
        self.window = window
        self.history = np.array([float()] * window)


class DNN(_Model):
    """Deep Neural Network
    based on `tf.estimator.DNNRegressor`
    """

    def _input_fn(self, X, y=None):
        """NumPy data to `tf.Tensor`.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector

        Returns
        -------
        features: dict
            Features tensor
        targets: tf.constant
            Targets tensor
        """
        if X.shape[1] != self.state.window:
            raise ValueError('%d=`X`.shape[1]!=state.window=%d' % (
                X.shape[1], self.state.window))
        features = {'t-%d' % (j + 1): tf.constant(xi)
                    for j, xi in enumerate(X)}
        # print(features)
        if y is not None:
            targets = tf.constant(y.reshape(-1, 1))
            print(targets)
            return features, targets
        return features

    def __init__(self, hidden_units, window, name='DNN'):
        """Constructs a `DNN` instance.

        Parameters
        ----------
        hidden_units: list
            DNN layers architecture
        window: int
            Window size
        name: str
            Model name
        """
        self.state = DNNState(window)
        self.name = name

        self.model = tf.estimator.DNNRegressor(
            hidden_units=hidden_units,
            feature_columns=[tf.feature_column.numeric_column(
                't-%d' % (j + 1)) for j in range(window)]
        )

    def fit(self, X, y, steps=100):
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
        def fit_input_fn():
            return self._input_fn(X, y)
        self.model.train(fit_input_fn, steps=steps)
        return self

    def predict(self, X):
        def predict_input_fn():
            return self._input_fn(X)
        return self.model.predict(predict_input_fn)

    def score(self, X, y):
        return 0

    def next(self):
        raise NotImplementedError
