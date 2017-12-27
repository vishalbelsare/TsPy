from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import tspy


class TestSetup(unittest.TestCase):

    NUM_POINTS = 500
    WINDOW = 10
    HIDDEN_UNITS = [50]
    NAME = 'dnn'
    NUM_EPOCHS = 5000

    ts = np.sin(np.linspace(-2 * np.pi, 2 * np.pi, NUM_POINTS))
    X_train, y_train = tspy.data.ar.Xy(ts, WINDOW)
    dnn = tspy.model.ar.DNN(
        hidden_units=HIDDEN_UNITS,
        window=WINDOW,
        name=NAME
    ).fit(X_train, y_train, num_epochs=NUM_EPOCHS)

    def test_predict(self):
        return self.assertIsInstance(self.dnn(self.X_train), np.ndarray)

    def test_score(self):
        return self.assertIsInstance(self.dnn(self.X_train, self.y_train), np.float32)

    def test_next(self):
        return self.assertIsInstance(self.dnn.next(10), np.ndarray)


if __name__ == '__main__':
    unittest.main()
