
import numpy as np
import time
from mtuq.util.util import iterable

try:
    import h5py
except:
    warn('Could not import h5py.')


def timer(func):
    """ Decorator for measuring execution time; prints elapsed time to
        standard output
    """
    def timed_func(*args, **kwargs):
        if getattr(kwargs, 'verbose', False):
            return func(*args, **kwargs)

        else:
            start_time = time.time()
            output = func(*args, **kwargs)
            _elapsed_time = time.time() - start_time
            print '  Elapsed time (s): %f\n' % _elapsed_time
            return output

    return timed_func


@timer
def grid_search_mt(data, greens, misfit, grid, verbose=True):
    """ Serial grid search 

    Grid search over moment tensors. For each moment tensor in grid, generates
    synthetics and evaluates data misfit
    """
    # creates an object we can iterate over
    zipped = zip(iterable(data), iterable(greens), iterable(misfit))

    results = np.zeros(grid.size)
    count = 0

    for mt in grid:
        print grid.index
        for _data, _greens, _misfit in zipped:
            results[count] += _misfit(_data, _greens, mt)
        count += 1

    return results


def grid_search_mt_origin(data, greens, misfit, grid):
    raise NotImplementedError


