from __future__ import absolute_import


def _Pipeline(*steps):
    """Pipeline transformations

    Parameters
    ----------
    steps: dict
        * transform: function
            Transformed to be applied
        * args: dict
            Arguments of the `transform`

    Returns
    -------
    data: numpy.ndarray | pandas.Series | pandas.DataFrame
        Output data after transforms
    """
    ret = None
    for step in steps:
        if ret is not None:
            ret = step['transform'](ret, **step['args'])
        else:
            ret = step['transform'](**step['args'])
    return ret
