from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import time
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 30, 10
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

import sklearn
# regressors
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, Lars
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# pipeline
from sklearn.pipeline import Pipeline
# validation
from sklearn.model_selection import GridSearchCV
# metrics
from sklearn.metrics import mean_squared_error
# LSTM
from lstm import LSTM

# base classes
from base import RollingWindow
from base import BackTest


class Model(object):
    """Predictive Model Class
    
    Examples
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from model import Model
    >>> ts = np.cumprod(np.random.normal(1.0, 0.03, 100))
    >>> split = len(ts) * 0.7
    >>> ts_train, ts_test = ts[:split], ts[split:]
    >>> model = Model(estimator='ridge', order=10)
    >>> model.fit(ts_train)
    >>> backtest = model.backtest(series_test)
    >>> backtest.plot()
    >>> plt.show()
    """
    
    def __init__(self, estimator='randomforest', order=5, **kwargs):
        """Constructor

        Parameters
        ----------
        estimator: str
            Estimator name
                * SVR (http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
                * Ridge (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
                * Lasso (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
                * Lars (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html)
                * ElasticNet (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
                * SGDRegressor (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
                * LSTM (https://keras.io/layers/recurrent/#lstm)
        order: int
            Autoregressive term order
        kwargs: keyword arguments
            Grid search named parameters

        TODO: compare trained model with ARIMA and Persitance model (x[t] = x[t-1])
              if MSE of the model is worse do not deploy, fallback to any of the two
        TODO: MSE := Model MSE
              MSE Baseline := Persistance Model MSE
              MSE ARIMA
              Calculate all at Backtest class, irrelevant to the model itself
        """
        self._order = self._validate_order(order)
        self._estimator, _params = self._validate_estimator_params(estimator, kwargs)
        _batch = isinstance(self._estimator, LSTM)
        if bool(_params):
            _grid = GridSearchCV(self._estimator, param_grid=_params)
            _regr = ('grid', _grid)
        else:
            # if no default or provided params, do not cross validate
            _regr = ('regressor', self._estimator)
        self.normaliser = kwargs.get('normaliser', self._normaliser) if kwargs.get('normalise', False) else None
        _rw = ('rolling', RollingWindow(window=self._order, batch=_batch, normaliser=self.normaliser))
        self._rw_bt = RollingWindow(window=self._order+1, batch=_batch, normaliser=self.normaliser)
        self.model = Pipeline([_rw, _regr])


    def _validate_estimator_params(self, estimator, kwargs):
        """Validate estimator and parameters inputs

        Parameters
        ----------
        estimator: str
            Estimator name
        kwargs: keyword arguments
            Grid search named parameters

        Returns
        -------
        estimator: sklearn.Estimator
            sklearn estimator, implementing `fit` and `predict`
        params: dict
            Grid search params
        
        TODO: think about default ranges for grid search
        """
        if not isinstance(estimator, str):
            raise TypeError('estimator argument must be str, but received %s' % type(estimator))
        _estimator = estimator.lower()
        if _estimator == 'svr':
            _kernel = kwargs.get('kernel', 'rbf')
            _C = kwargs.get('C', np.logspace(-4, 4, 5))
            _epsilon = kwargs.get('epsilon', np.logspace(-4, 4, 5))
            _gamma = kwargs.get('gamma', 'auto')
            _degree = kwargs.get('degree', 3)
            return SVR(kernel=_kernel, degree=_degree), {'C': _C, 'epsilon': _epsilon}
        if _estimator == 'ridge':
            _alpha = kwargs.get('alpha', np.logspace(-4, 4, 20))
            return Ridge(), {'alpha': _alpha}
        if _estimator == 'lasso':
            _alpha = kwargs.get('alpha', np.logspace(-4, 4, 20))
            return Lasso(), {'alpha': _alpha}
        if _estimator == 'lars':
            _n_nonzero_coefs = kwargs.get('n_nonzero_coefs', np.inf)
            return Lars(), {'n_nonzero_coefs': _n_nonzero_coefs}
        if _estimator == 'elasticnet':
            _alpha = kwargs.get('alpha', np.logspace(-4, 4, 20))
            return ElasticNet(), {'alpha': _alpha}
        if _estimator == 'sgd' or _estimator == 'sgdregressor':
            _alpha = kwargs.get('alpha', np.logspace(-4, 4, 20))
            return SGDRegressor(), {'alpha': _alpha}
        if _estimator == 'randomforest':
            _n_estimators = range(5, 30, 5)
            return RandomForestRegressor(), {'n_estimators': _n_estimators}
        if _estimator == 'adaboost':
            _n_estimators = range(10, 60, 5)
            _learning_rate = np.logspace(-2, 1, 4)
            return AdaBoostRegressor(), {'n_estimators': _n_estimators, 'learning_rate': _learning_rate}
        if _estimator == 'gradientboosting':
            _n_estimators = range(10, 60, 5)
            _learning_rate = np.logspace(-2, 1, 4)
            return GradientBoostingRegressor(), {'n_estimators': _n_estimators, 'learning_rate': _learning_rate}
        if _estimator == 'lstm':
            _layers = kwargs.get('layers', [1, self._order, 2*self._order, 1])
            _pct_dropout = kwargs.get('pct_dropout', 0.5)
            return LSTM(layers=_layers, pct_dropout=_pct_dropout), {}

    def _validate_order(self, order):
        """Validate autoregressive order parameter
        
        Parameters
        ----------
        order: int
            Autoregressive term order
        
        Returns
        -------
        order: int
            Autoregressive term order
        """
        if not isinstance(order, int):
            raise TypeError('order argument must be int, but received %s' % type(order))
        else:
            return order


    def fit(self, series, **kwargs):
        """Train the model
        
        Parameters
        ----------
        series: pandas.Series
            Time series to train the model on
        """
        _X, _y = self._format(series)
        # sklearn.pipeline.Pipeline docs (http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
        # fit parameters are passed by prefixing their names by the name of the step
        # example: model = Pipeline([('rolling', RollingWindow(5)), ('regressor', LSTM([1, 5, 10, 1]))])
        # model.fit(X, y, regressor__epochs=5)
        _prefix = self.model.steps[-1][0]
        _kwargs = { '%s__%s' % (_prefix, key): value for key, value in kwargs.items() }
        self.model.fit(_X, _y, **_kwargs)


    def _format(self, series):
        """Data formatter
        
        Parameters
        ----------
        series: pandas.Series
            Time series to format
        
        Returns
        -------
        series: pandas.Series
            Original time series
        targets: numpy.ndarray
            Targets vector for supervised learning
        """
        return series, series[self._order:].values.reshape((-1, 1)).ravel()

    def _normaliser(self, series):
        """Normaliser for each (window) sub-series
        
        Parameters
        ----------
        series: pandas.Series
            Time series to be normalised
        
        Returns
        -------
        series: pandas.Series
            Normalised time series
        """
        return (series / series[0]) - 1


    def predict(self, series):
        """Prediction using provided series
        
        """
        return self.model.predict(series).ravel()

    def backtest(self, series):
        """Backtest model to the provided series
        
        Parameters
        ----------
        series: pandas.Series
            Time series to be backtested

        Returns
        -------
        backtest: base.BackTest
            Backtest object, implements methods: `summary`, `plot`, `outliers`
        """
        series_hat = self.predict(series)[:-1]
        _norm = self._rw_bt.transform(series)[:, -1]
        _series = pd.Series(_norm, index=series.index[-len(_norm):])
        return BackTest(_series, series_hat)