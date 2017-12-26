from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np

import tspy


def FinanceXy(ticker, window, start_date=None, end_date=None):
    """Finance AR supervised data

    Parameters
    ----------
    ticker: str
        Ticker name
    window: int
        Window size

    Returns
    -------
    X: numpy.ndarray
        Features matrix
    y: numpy.ndarray
        Target vector
    """
    _finance = {'transform': tspy.adapter.Finance.Prices,
                'args': {'tickers': [ticker], 'start_date': start_date, 'end_date': end_date}}
    _Xy = {'transform': tspy.data.ar.Xy,
           'args': {'window': window}}
    return tspy.framework.pipeline._Pipeline(_finance, _Xy)
