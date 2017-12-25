from __future__ import absolute_import

# scientific computing
import numpy as np
import tensorflow as tf
import pandas as pd


class _Model:

    def __init__(self, state=None, name=None):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError
