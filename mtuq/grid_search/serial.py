
import numpy as np
import time

try:
    import h5py
except:
    warn('Could not import h5py.')


def timer(func):
    """ Decorator for measuring execution time; prints elapsed time to
        standard output
    """
    def timed_func(*args, **kwargs):
        start_time = time.time()

        output = func(*args, **kwargs)

        _elapsed_time = time.time() - start_time
        print '  Elapsed time (s): %f\n' % _elapsed_time

        return output

    return timed_func


@timer
def grid_search_serial(data, greens, misfit, grid):
    """ Serial grid search 

    Grid search over moment tensors. For each moment tensor in grid, generates
    synthetics and evaluates data misfit
    """
    results = np.zeros(grid.size)
    count = 0

    for mt in grid:
        print grid.index
        for key in data:
            results[count] += misfit[key](data[key], greens[key], mt)
        count += 1

    return results

