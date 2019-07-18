
import numpy as np
import time
from mtuq.util import timer

try:
    import h5py
except:
    warn('Could not import h5py.')


@timer
def grid_search_mt(data, greens, misfit, grid, verbose=True):
    """ Serial grid search over moment tensors
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
    """ Serial grid search over mechanism and location parameters
    """
    results = np.zeros((len(sources), len(origins)))

    # carry out search
    for _i, origin in enumerate(origins):
        for _j, mt in enumerate(sources):

            if verbose and not ((_i*npts_inner+_j) % np.ceil(0.1*npts_outer)):
                print _message(_i*npts_inner+_j, npts_outer)

            results[_i, _j] = misfit(data, greens.subset(origin), source)

    return results



def _message(ii, nn):
    return (
            '  about %2d%% finished\n'
            % np.ceil((100.*ii/nn))
           )

