from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np
import pandas as pd

# market data bundler
import pandas_datareader.data as web


class Finance:
    """Market Data Wrapper."""
    start_date = None
    end_date = None
    source = 'quandl'
    _close_col = {'quandl': 'Close', 'yahoo': 'Adj Close'}

    @classmethod
    def _get(cls, ticker, **kwargs):
        """Helpder method for `web.DataReader`.
        Parameters
        ----------
        ticker: str
            Ticker name
        **kwargs: dict
            Arguments for `quandl.get`
        Returns
        -------
        df: pandas.DataFrame
            Table of prices for `ticker`
        """
        return web.DataReader(ticker, cls.source, **kwargs)

    @classmethod
    def Returns(cls, tickers):
        """Get daily returns for `tickers`.
        Parameters
        ----------
        tickers: list
            List of ticker names
        Returns
        -------
        df: pandas.DataFrame
            Table of Returns of Adjusted Close prices for `tickers`
        """
        return cls.Prices(tickers).pct_change()[1:]

    @classmethod
    def Prices(cls, tickers):
        """Get daily prices for `tickers`.
        Parameters
        ----------
        tickers: list
            List of ticker names
        Returns
        -------
        df: pandas.DataFrame
            Table of Adjusted Close prices for `tickers`
        """
        return pd.DataFrame.from_dict({ticker: cls._get(ticker, start=cls.start_date,
                                                        end=cls.end_date)[cls._close_col[cls.source]] for ticker in tickers})
