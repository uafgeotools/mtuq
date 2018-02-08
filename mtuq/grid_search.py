
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.math import PI, INF
from mtuq.util.util import Struct

from copy import deepcopy


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




### moment tensor grids

class Grid(object):
    """ Multidimensional grid

        Allows iterating over all values of a multidimensional grid while 
        storing only values along the axes

        Given a set of axes names and values, __init__ can be called directly
        to return a Grid instance, or __init__ can be overloaded to create a
        specialized Grid subclass
    """
    def __init__(self, axes, map=None):

        # dictionary containing axis names and values
        self.axes = axes

        # optional map from one parameterization to another
        self.map = map

        # what is the length along each axis?
        shape = []
        for _, axis in axes.items():
            shape += [len(axis)]

        # what attributes would the grid have if stored as an numpy array?
        self.shape = shape
        self.size = np.product(shape)
        self.ndim = len(shape)
 

    def get(self, i):
        """ Returns i-th point of grid
        """
        p = Struct()
        for key, val in self.axes.items():
            p[key] = val[i%len(val)]
            i/=len(val)

        if self.map:
            return self.map(p)
        else:
            return p


class MTGridRandom(Grid):
    """ Full moment tensor grid with randomly-spaced values
    """
    def __init__(self, Mw=[], points_per_axis=10):
        N = points_per_axis

        # upper bound, lower bound, number of points
        v = [-1./3., 1./3., N]
        w = [-3./8.*PI, 3./8.*PI, N]
        kappa = [0., 360, N]
        sigma = [-90., 90., N]
        h = [0., 1., N]

        # magnitude is treated separately
        rho = _array(Mw)/np.sqrt(2)

        super(MTGridRandom, self).__init__({
            'rho': rho,
            'v': randvec(v),
            'w': randvec(w),
            'kappa': randvec(kappa),
            'sigma': randvec(sigma),
            'h': randvec(h)},
            tape2015)


class MTGridRegular(Grid):
    """ Full moment tensor grid with regularly-spaced values
    """
    def __init__(self, Mw=[], points_per_axis=10):
        N = points_per_axis

        # upper bound, lower bound, number of points
        v = [-1./3., 1./3., N]
        w = [-3./8.*PI, 3./8.*PI, N]
        kappa = [0., 360, N]
        sigma = [-90., 90., N]
        h = [0., 1., N]

        # magnitude is treated separately
        rho = _array(Mw)/np.sqrt(2)

        super(MTGridRandom, self).__init__({
            'rho': rho,
            'v': linspace(v),
            'w': linspace(w),
            'kappa': linspace(kappa),
            'sigma': linspace(sigma),
            'h': linspace(n)},
            tape2015)


class DCGridRandom(Grid):
    """ Double-couple moment tensor grid with randomly-spaced values
    """
    def __init__(self, Mw=[], points_per_axis=10):
        N = points_per_axis

        # upper bound, lower bound, number of points
        kappa = [0., 360, N]
        sigma = [-90., 90., N]
        h = [0., 1., N]

        # magnitude is treated separately
        rho = _array(Mw)/np.sqrt(2)

        super(DCGridRandom, self).__init__({
            'rho': rho,
            'v': np.array([0.]),
            'w': np.array([0.]),
            'kappa': randvec(kappa),
            'sigma': randvec(sigma),
            'h': randvec(h)},
            tape2015)


class DCGridRegular(Grid):
    """ Double-couple moment tensor grid with regularly-spaced values
    """
    def __init__(self, Mw=[], points_per_axis=10):
        N = points_per_axis

        # upper bound, lower bound, number of points
        kappa = [0., 360, N]
        sigma = [-90., 90., N]
        h = [0., 1., N]

        # magnitude is treated separately
        rho = _array(Mw)/np.sqrt(2)

        super(DCGridRegular, self).__init__({
            'rho': rho,
            'v': np.array([0.]),
            'w': np.array([0.]),
            'kappa': linspace(kappa),
            'sigma': linspace(sigma),
            'h': linspace(n)},
            tape2015)


def tape2015(p):
    return tt152cmt(p.rho, p.v, p.w, p.kappa, p.sigma, p.h)




### depth/location grids

def DepthGrid():
    raise NotImplementedError


def OriginGrid():
    raise NotImplementedError



### utilities

def _array(x):
    if type(x) in [np.ndarray]:
        return x

    elif type(x) in [list,tuple]:
        return np.array(x)

    elif type(x) in [float,int]:
        return np.array([float(x)])

    else:
        raise ValueError


def linspace(x1,x2,nx):
    return np.linspace(x1,x2,n+2)[1:-1]


def randvec(args):
    return np.random.uniform(*args)


