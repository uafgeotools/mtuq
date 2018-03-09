
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.math import PI, INF
from mtuq.util.util import Struct

from numpy.random import uniform as random
from mtuq.util.math import open_interval as regular



class Grid(object):
    """ Structured grid

    Allows iterating over all values of a multidimensional grid while 
    storing only values along the axes

    param dict: dictionary containing names of parameters and values along
        corresponding axes
    """
    def __init__(self, dict, start=0, stop=None, callback=None):

        # list of parameter names
        self.keys = dict.keys()
        
        # corresponding list of axis arrays
        self.vals = dict.values()

        # what is the length along each axis?
        shape = []
        for axis in self.vals:
            shape += [len(axis)]

        # what attributes would the grid have if stored as a numpy array?
        self.ndim = len(shape)
        self.shape = shape

        # what part of the grid do we want to iterate over?
        self.start = start
        if stop:
            self.stop = stop
            self.size = stop-start
        else:
            self.stop = np.product(shape)
            self.size = np.product(shape)-start

        self.index = start

        # optional map from one parameterization to another
        self.callback = callback

 
    def get(self, i):
        """ Returns i-th point of grid
        """
        p = Struct()
        for key, val in zip(self.keys, self.vals):
            p[key] = val[i%len(val)]
            i/=len(val)

        if self.callback:
            return self.callback(p)
        else:
            return p


    def decompose(self, nproc):
        """ Decomposes grid for parallel processing
        """
        if self.start!=0:
            raise Exception

        subsets = []
        for iproc in range(nproc):
            start=iproc*self.size/nproc
            stop=(iproc+1)*self.size/nproc
            items = zip(self.keys, self.values)
            subsets += [Grid(dict(items), start, stop, callback=self.callback)]
        return subsets


    def save(self, filename, dict):
        """ Saves a set of values defined on grid
        """
        raise NotImplementedError


    # the next two methods make it possible to iterate over the grid
    def next(self): 
        if self.index < self.stop:
           # get the i-th point in grid
           p = self.get(self.index)
        else:
            raise StopIteration
        self.index += 1
        return p


    def __iter__(self):
        return self




class UnstructuredGrid(object):
    """ Unstructured grid

    param dict: dictionary containing the complete set of coordinate values for
       each parameter
    """
    def __init__(self, dict, start=0, stop=None, callback=None):

        # list of parameter names
        self.keys = dict.keys()

        # corresponding list of coordinate arrays
        self.vals = dict.values()

        # there is no shape attribute because it is an unstructured grid,
        # however, ndim and size are still well defined
        self.ndim = len(self.vals)
        size = len(self.vals[0])

        # check consistency
        for array in self.vals:
            assert len(array) == size

        # what part of the grid do we want to iterate over?
        self.start = start
        if stop:
            self.stop = stop
            self.size = stop-start
        else:
            self.stop = size
            self.size = size-start

        self.index = start

        # optional map from one parameterization to another
        self.callback = callback


    def get(self, i):
        """ Returns i-th point of grid
        """
        i -= self.start
        p = Struct()
        for key, val in zip(self.keys, self.vals):
            p[key] = val[i]

        if self.callback:
            return self.callback(p)
        else:
            return p


    def decompose(self, nproc):
        """ Decomposes grid for parallel processing
        """
        subsets = []
        for iproc in range(nproc):
            start=iproc*self.size/nproc
            stop=(iproc+1)*self.size/nproc
            dict = {}
            for key, val in zip(self.keys, self.vals):
                dict[key] = val[start:stop]
            subsets += [UnstructuredGrid(dict, start, stop, callback=self.callback)]
        return subsets


    def save(self, filename, dict):
        """ Saves a set of values defined on grid
        """
        raise NotImplementedError


    # the next two methods make it possible to iterate over the grid
    def next(self): 
        if self.index < self.stop:
           # get the i-th point in grid
           p = self.get(self.index)
        else:
            raise StopIteration
        self.index += 1
        return p


    def __iter__(self):
        return self



def MTGridRandom(Mw=[], npts=500000):
    """ Full moment tensor grid with randomly-spaced values
    """
    N = npts

    # upper bound, lower bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = float(Mw)/np.sqrt(2)

    return UnstructuredGrid({
        'rho': rho*np.ones(N),
        'v': random(*v),
        'w': random(*w),
        'kappa': random(*kappa),
        'sigma': random(*sigma),
        'h': random(*h)},
        callback=tape2015)


def MTGridRegular(Mw, npts_per_axis=25):
    """ Full moment tensor grid with regularly-spaced values
    """
    N = npts_per_axis

    # upper bound, lower bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = cast(Mw)/np.sqrt(2)

    return Grid({
        'rho': rho,
        'v': regular(*v),
        'w': regular(*w),
        'kappa': regular(*kappa),
        'sigma': regular(*sigma),
        'h': regular(*h)},
        callback=tape2015)


def DCGridRandom(Mw, npts=50000):
    """ Double-couple moment tensor grid with randomly-spaced values
    """
    N = npts

    # upper bound, lower bound, number of points
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = float(Mw)/np.sqrt(2)

    return UnstructuredGrid({
        'rho': rho*np.ones(N),
        'v': np.zeros(N),
        'w': np.zeros(N),
        'kappa': random(*kappa),
        'sigma': random(*sigma),
        'h': random(*h)},
        callback=tape2015)


def DCGridRegular(Mw, npts_per_axis=25):
    """ Double-couple moment tensor grid with regularly-spaced values
    """ 
    N = npts_per_axis

    # upper bound, lower bound, number of points
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = cast(Mw)/np.sqrt(2)

    return Grid({
        'rho': rho,
        'v': np.array([0.]),
        'w': np.array([0.]),
        'kappa': regular(*kappa),
        'sigma': regular(*sigma),
        'h': regular(*h)},
        callback=tape2015)


def DepthGrid():
    raise NotImplementedError


def OriginGrid():
    raise NotImplementedError



### utilities

def cast(x):
    if type(x) in [np.ndarray]:
        return x
    elif type(x) in [list, tuple]:
        return np.array(x)
    elif type(x) in [float, int]:
        return np.array([float(x)])
    else:
        raise ValueError


def tape2015(p):
    return tt152cmt(p.rho, p.v, p.w, p.kappa, p.sigma, p.h)

