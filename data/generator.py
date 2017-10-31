import numpy as np
import pandas as pd
import datetime

def randomWalk(N=1000, name='ts', end=datetime.date.today(), freq=1):
    """
    Single random series generator.

    Parameters
    ----------
    N: int
        Number of datapoints
    name: str
        Name of the generated series
    end: datetime.date
        End date of the series
    freq: datetime.timedelta
        Sampling freq

    Returns
    -------
    series: pandas.Series
        Generated series
    """
    return pd.Series(np.random.uniform(333, 999) * np.cumprod(np.random.normal(1, 0.03, N)), name=name, index=pd.date_range(end=end, freq=str(freq) + 'B', periods=N))