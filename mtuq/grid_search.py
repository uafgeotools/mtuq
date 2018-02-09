
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.math import PI, INF
from mtuq.util.util import Struct


def grid_search(data, greens, misfit, grid):
    """ Grid search over moment tensor parameters
    """
    best_misfit = INF

    for _i in range(grid.size):
        # gets the i-th moment tensor in grid
        print _i
        mt = grid.get(_i)

        # generate_synthetics
        categories = data.keys()
        synthetics = {}
        for key in categories:
            synthetics[key] = greens[key].get_synthetics(mt)

        sum_misfit = 0.
        for key in categories:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            sum_misfit += chi(dat, syn)

        # keep track of best moment tensor
        if sum_misfit < best_misfit:
            best_mt = mt


def grid_search_mpi(data, greens, misfit, grid):
    raise NotImplementedError



