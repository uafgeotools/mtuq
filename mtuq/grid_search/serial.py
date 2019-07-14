
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
def grid_search_mt_depth(data, greens, misfit, grid, depths, verbose=True):
    """ Serial grid search over moment tensors and depths
    """
    assert isinstance(greens, dict), TypeError
    assert hasattr(misfit, '__call__'), TypeError

    npts_inner = grid.size
    npts_outer = grid.size*len(depths)
    results = {}
    for depth in depths:
        results[depth] = np.zeros(npts_inner)

    # carry out search
    for _i, depth in enumerate(depths):
        for _j, mt in enumerate(grid):

            if verbose and not ((_i*npts_inner+_j) % np.ceil(0.1*npts_outer)):
                print _message(_i*npts_inner+_j, npts_outer)

            results[depth][_j] = misfit(data, greens[depth], mt)

    return results



def _message(ii, nn):
    return (
            '  about %2d%% finished\n'
            % np.ceil((100.*ii/nn))
           )

