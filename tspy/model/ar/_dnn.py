from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import tensorflow as tf

from tspy.model.ar._base import _ARModel


class DNN(_ARModel):
    """Deep Neural Network
    based on `tf.estimator.DNNRegressor`
    """

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
        super(DNN, self).__init__(window, name)

        self.model = tf.estimator.DNNRegressor(
            hidden_units=hidden_units,
            feature_columns=[tf.feature_column.numeric_column(
                't-%d' % (j + 1)) for j in range(self.state.window)]
        )

    def _input_fn(self, X, y=None, num_epochs=1):
        """NumPy data to `tf.Tensor`.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector

        Returns
        -------
        input_fn: tf.estimator.inputs.numpy_input_fn
            Input function for estimator
        """
        if X.shape[1] != self.state.window:
            print(X.shape[1])
            print(self.state.window)
            raise ValueError('%d=`X`.shape[1]!=state.window=%d' % (
                X.shape[1], self.state.window))
        features = {'t-%d' % (j + 1): xi
                    for j, xi in enumerate(X.T)}
        if y is not None:
            targets = y
            return tf.estimator.inputs.numpy_input_fn(x=features, y=targets, num_epochs=num_epochs, shuffle=False)
        return tf.estimator.inputs.numpy_input_fn(x=features, shuffle=False)

    def _fit(self, X, y, num_epochs=1):
        """Model fitting.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector
        num_epochs: int
            Number of epochs
        """
        self.model.train(self._input_fn(X, y, num_epochs),
                         steps=int(num_epochs / 10))

    def _predict(self, X):
        """Predict method.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix

        Returns
        -------
        y_hat: numpy.ndarray | NoneType
            Predicted target vector
        """
        return np.array([y['predictions'] for y in self.model.predict(self._input_fn(X))])

    def _score(self, X, y):
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
        return float(self.model.evaluate(self._input_fn(X, y, 1), steps=None)['loss'])
