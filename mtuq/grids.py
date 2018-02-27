
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.math import PI, INF
from mtuq.util.util import Struct


class Grid(object):
    """ Structured grid

        Allows iterating over all values of a multidimensional grid while 
        storing only values along the axes
    """
    def __init__(self, dict, start=0, stop=None, map=None):

        # list of parameter names
        self.keys = dict.keys()
        
        # corresponding list of parameter arrays
        self.vals = dict.values()

        # what is the length along each axis?
        shape = []
        for axis in self.vals:
            shape += [len(axis)]
        self.ndim = len(shape)

        # are we considering only a subset of the grid?
        if stop:
            self.size = stop
        else:
            self.size = np.product(shape)
        if start:
            self.size -= start

        self.start = start

        # optional map from one parameterization to another
        self.map = map

 
    def get(self, i):
        """ Returns i-th point of grid
        """
        i += self.start
        p = Struct()
        for key, val in zip(self.keys, self.vals):
            p[key] = val[i%len(val)]
            i/=len(val)

        if self.map:
            return self.map(p)
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
            subsets += [Grid(self.dict(), start, stop, map=self.map)]
        return subsets


    def save(self, filename, array):
        """ Saves a set of values defined on grid
        """
        raise NotImplementedError


    def dict(self):
        return dict(zip(self.keys, self.vals))



class UnstructuredGrid(object):
    """ Unstructured grid
    """
    def __init__(self, dict, start=0, stop=None, map=None):

        # list of parameter names
        self.keys = dict.keys()

        # corresponding list of parameter arrays
        self.vals = dict.values()

        # check consistency
        npts = []
        for array in self.vals:
            npts += [len(array)]
        self.ndim = len(npts)

        # are we considering only a subset of the grid?
        if stop:
            size = stop
        else:
            size = npts[0]
        if start:
            size -= start

        self.size = size
        self.start = start

        # optional map from one parameterization to another
        self.map = map


    def get(self, i):
        """ Returns i-th point of grid
        """
        p = Struct()
        for key, val in zip(self.keys, self.vals):
            p[key] = val[i]

        if self.map:
            return self.map(p)
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
            subsets += [UnstructuredGrid(dict, start, stop, map=self.map)]
        return subsets


    def save(self, filename, array):
        """ Saves a set of values defined on grid
        """
        raise NotImplementedError




class MTGridRandom(UnstructuredGrid):
    """ Full moment tensor grid with randomly-spaced values
    """
    def __init__(self, Mw=[], npts=500000):
        N = npts

        # upper bound, lower bound, number of points
        v = [-1./3., 1./3., N]
        w = [-3./8.*PI, 3./8.*PI, N]
        kappa = [0., 360, N]
        sigma = [-90., 90., N]
        h = [0., 1., N]

        # magnitude is treated separately
        rho = float(Mw)/np.sqrt(2)

        super(MTGridRandom, self).__init__({
            'rho': rho*np.ones(N),
            'v': randvec(v),
            'w': randvec(w),
            'kappa': randvec(kappa),
            'sigma': randvec(sigma),
            'h': randvec(h)},
            map=tape2015)


class MTGridRegular(Grid):
    """ Full moment tensor grid with regularly-spaced values
    """
    def __init__(self, Mw=[], npts_per_axis=10):
        N = npts_per_axis

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
            map=tape2015)


class DCGridRandom(UnstructuredGrid):
    """ Double-couple moment tensor grid with randomly-spaced values
    """
    def __init__(self, Mw=[], npts=50000):
        N = npts

        # upper bound, lower bound, number of points
        kappa = [0., 360, N]
        sigma = [-90., 90., N]
        h = [0., 1., N]

        # magnitude is treated separately
        rho = float(Mw)/np.sqrt(2)

        super(DCGridRandom, self).__init__({
            'rho': rho*np.ones(N),
            'v': np.zeros(N),
            'w': np.zeros(N),
            'kappa': randvec(kappa),
            'sigma': randvec(sigma),
            'h': randvec(h)},
            map=tape2015)


class DCGridRegular(Grid):
    """ Double-couple moment tensor grid with regularly-spaced values
    """
    def __init__(self, Mw=[], npts_per_axis=20):
        N = npts_per_axis

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
            map=tape2015)


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

def tape2015(p):
    return tt152cmt(p.rho, p.v, p.w, p.kappa, p.sigma, p.h)

