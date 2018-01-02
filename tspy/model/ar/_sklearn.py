from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor

from tspy.model import _Model
from tspy.model.ar._state import ARState


class Regressor(_Model):
    """`scikit-learn` regressors wrapper class`
    """

    def __init__(self, regressor_type, window, name='Regressor', **regressor_kwargs):
        """Constructs a `Regressor` instance.

        Parameters
        ----------
        regressor_type: str
            Type of regressor
        window: int
            Window size
        name: str
            Model name
        regressor_kwargs: dict
            Arguments of `scikit-learn` regressor
        """
        super(Regressor, self).__init__(name)
        self.state = ARState(window)

        self.model = self._parse_regressor(regressor_type, **regressor_kwargs)

    def _parse_regressor(self, regressor_type, **kwargs):
        """Regressor type parser.

        Parameters
        ----------
        regressor_type: str
            Type of regressor
        kwargs: dict
            Regressor-specific arguments

        Returns
        -------
        regressor: object
            `scikit-learn` regressor
        """
        if regressor_type.lower() == 'ridge':
            return Ridge(**kwargs)
        elif regressor_type.lower() == 'lasso':
            return Lasso(**kwargs)
        elif regressor_type.lower() == 'gaussian process':
            return GaussianProcessRegressor(**kwargs)

    def _fit(self, X, y):
        """Model fitting.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix
        y: numpy.ndarray | NoneType
            Target vector
        """
        self.model.fit(X, y)

    def _set_state(self, X, y):
        print('yo')
        self.state.history = np.append(X[-1, :-1], y[-1])

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
        return self.model.predict(X)

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
        return self.model.score(X, y)

    def next(self, steps=1):
        _next = []
        for i in range(steps):
            _next.append(self.predict(self.state.history))
        return np.array(_next).reshape(steps, 1)
