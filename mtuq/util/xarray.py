
import numpy as np


def dataarray_to_table(da, dims):
    values = da.values.flatten()

    stacked = da.stack(z=dims)
    indexes = stacked.indexes['z']

    table = np.empty((da.size, 3))
    for _i, vw in enumerate(indexes):
        table[_i, 0] = vw[0]
        table[_i, 1] = vw[1]
        table[_i, 2] = values[_i]

    return table


def array_to_dict(array, shape):
    if array.ndim == 1:
        array.reshape((array.size, 1))

    # how many sets of values are contained in array?
    nout = array.shape[1]

    if array.shape[0] != np.product(shape):
        raise ValueError("Mismatch between array and grid shape")

    if nout > 1:
        keys = ['%06d' % _i for  _i in range(nout)]
    else:
        keys = ['']

    return {keys[_i]: array[:,_i] for _i in range(nout)}

