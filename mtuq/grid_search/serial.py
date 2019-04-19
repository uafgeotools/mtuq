
import numpy as np
import time
from mtuq.util import timer

try:
    import h5py
except:
    warn('Could not import h5py.')


@timer
def grid_search_mt(data_list, greens_list, misfit_list, grid, verbose=True):
    """ Serial grid search over moment tensors
    """
    # create iterator
    zipped = zip(data_list, greens_list, misfit_list)

    # initialize results
    npts = grid.size
    results = np.zeros(npts)

    # carry out search
    for _i, mt in enumerate(grid):
        if verbose and not(_i % np.ceil(0.1*npts)):
            print _message(_i, npts)

        for data, greens, misfit in zipped:
            results[_i] += misfit(data, greens, mt)

    return results



@timer
def grid_search_mt_depth(data_list, greens_list, misfit_list, grid, depths, verbose=True):
    """ Serial grid search over moment tensors and depths
    """

    for greens in greens_list:
        # Green's functions are depth dependent
        assert isinstance(greens, dict), TypeError

    for misfit in misfit_list:
        assert hasattr(misfit, '__call__'), TypeError

    # create iterator
    zipped = zip(data_list, greens_list, misfit_list)

    # initialize results
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

            for data, greens, misfit in zipped:
                results[depth][_j] += misfit(data, greens[depth], mt)

    return results



def _message(ii, nn):
    return (
            '  about %2d%% finished\n'
            % np.ceil((100.*ii/nn))
           )

