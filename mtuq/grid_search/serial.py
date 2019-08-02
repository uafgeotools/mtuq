
import numpy as np
import time
from mtuq.util import iterable, timer


@timer
def grid_search(data, greens, misfit, origins, sources, verbose=True):
    """ Evaluates misfit over origin and source grids (serial implementation)
    """
    results = []
    for origin in iterable(origins):
        results += [misfit(data, greens.select(origin), sources)]
    return np.concatenate(results)


