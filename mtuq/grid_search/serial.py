
import numpy as np
import time
from mtuq.util import timer

try:
    import h5py
except:
    warn('Could not import h5py.')


@timer
def grid_search_mt(data, greens, misfit, grid, verbose=True):
    """ Grid search over moment tensors
    """
    results = np.zeros(grid.size)

    # carry out search
    for _i, mt in enumerate(grid):
        if verbose and not(_i % np.ceil(0.1*grid.size)):
            print _message(_i, grid.size)

        results[_i] = misfit(data, greens, mt)

    return results



@timer
def grid_search(data, greens, misfit, sources, origins, verbose=True):
    """ Grid search over source mechanism and location parameters
    """
    ni, nj = len(origins), len(sources)
    results = np.zeros((ni, nj))

    # carry out search
    for _i, origin in enumerate(origins):
        for _j, source in enumerate(sources):

            if verbose and not ((_i*nj+_j) % np.ceil(0.1*ni*nj)):
                print _message(_i*nj+_j, ni*nj)

            results[_i, _j] = misfit(data, greens.subset(origin), source)

    return results



def _message(top,bot):
    return (
            '  about %2d%% finished\n'
            % np.ceil((100.*top/bot))
           )

