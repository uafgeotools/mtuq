
import numpy as np
import time
from mtuq.util import iterable, timer


@timer
def grid_search(data, greens, misfit, origins, sources, verbose=True):
    """ Grid search over source mechanism and location parameters
    """
    origins = iterable(origins)
    ni, nj = len(origins), len(sources)
    results = np.zeros((ni, nj))

    for _i, origin in enumerate(origins):
        _greens = greens.select(origin)
        for _j, source in enumerate(sources):
            results[_i, _j] = misfit(data, _greens, source)
            if verbose: 
                progress(_i,_j,ni,nj)

    return results



def progress(_i,_j,ni,nj):
    if ((_i*nj+_j) % np.ceil(0.1*ni*nj)):
        return
    print\
        '  about %2d%% finished\n'\
        % np.ceil((100.*(_i*nj+_j)/(ni*nj)))

