from __future__ import absolute_import
from __future__ import division

# scientific computing
import numpy as np


class ARState:
    """AR Model State."""

    def __init__(self, window):
        """Constructs a `ARState` instance.

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
