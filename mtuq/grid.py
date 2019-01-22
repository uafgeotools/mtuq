

import numpy as np
from mtuq.util.math import PI
from mtuq.util.util import asarray

from mtuq.util.moment_tensor import tape2015
from numpy.random import uniform as random
from mtuq.util.math import open_interval as regular

import warnings
import numpy as np

from mtuq.util.util import AttribDict, warn



class Grid(object):
    """ Structured grid

    Allows iterating over all values of a rectangular structured grid while 
    storing only values along the axes

    :param: dict: dictionary containing names of axes and corresponding lists
       of values along axes
    :param: start: when iterating over the grid, start at this element
    :param: stop: when iterating over the grid, stop at this element
    :param: callback: optional function applied to each grid point
       through a callback to the ``get`` method. Can be used to carry out a
       coordinate transformation or a more general reparameterizatoin.


    .. rubric:: Examples

    To cover the unit square with a N-by-N rectangular grid:

    .. code ::

       grid = Grid({'x': np.linspace(0., 1., N),
                    'y': np.linpsace(0., 1., N)})


    To parameterize the surface of the Earth with an N-by-2N Mercator grid:

    .. code::          

       grid = Grid({'latitude': np.linspace(-90., 90., N),
                    'longitude': np.linspace(-180., 180., 2*N)})


    .. rubric:: Iterating over grids

    The order of iteration over a grid is determined by the order of keys in
    the dictionary input argument.

    In the unit square example above, ``"x"`` is the fast axis and 
    y"`` is the slow axis.

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

        self.callback = callback

 
    def get(self, i):
        """ Returns i-th point of grid
        """
        p = AttribDict()
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
        import h5py
        with h5py.File(filename, 'w') as hf:
            for key, val in zip(self.keys, self.vals):
                hf.create_dataset(key, data=val)

            for key, val in dict.iteritems():
                hf.create_dataset(key, data=val)


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

    .. rubric:: Examples

   Unstructured grid consisting of N randomly-chosen points within the unit 
   square:

   .. code ::

      grid = UnstructuredGrid({'x': np.random.rand(N),
                               'y': np.random.rand(N)})

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
        p = AttribDict()
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
        import h5py
        with h5py.File(filename, 'w') as hf:
            for key, val in zip(self.keys, self.vals):
                hf.create_dataset(key, data=val)

            for key, val in dict.iteritems():
                hf.create_dataset(key, data=val)


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


def FullMomentTensorGridRandom(moment_magnitude=None, npts=50000):
    """ Full moment tensor grid with randomly-spaced values
    """
    N = npts

    if not moment_magnitude:
        raise ValueError
    Mw = moment_magnitude

    # lower bound, upper bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    M0 = 10.**(1.5*float(Mw) + 9.1)
    rho = M0*np.sqrt(2.)

    return UnstructuredGrid({
        'rho': rho*np.ones(N),
        'v': random(*v),
        'w': random(*w),
        'kappa': random(*kappa),
        'sigma': random(*sigma),
        'h': random(*h)},
        callback=tape2015_to_Mij)


def FullMomentTensorGridRegular(moment_magnitude=None, npts_per_axis=25):
    """ Full moment tensor grid with regularly-spaced values
    """
    N = npts_per_axis

    if not moment_magnitude:
        raise ValueError
    Mw = moment_magnitude

    # lower bound, upper bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    M0 = 10.**(1.5*float(Mw) + 9.1)
    rho = asarray(Mw)/np.sqrt(2)

    return Grid({
        'rho': asarray(rho),
        'v': regular(*v),
        'w': regular(*w),
        'kappa': regular(*kappa),
        'sigma': regular(*sigma),
        'h': regular(*h)},
        callback=tape2015_to_Mij)


def DoubleCoupleGridRandom(moment_magnitude=None, npts=50000):
    """ Double-couple moment tensor grid with randomly-spaced values
    """
    N = npts

    if not moment_magnitude:
        raise ValueError
    Mw = moment_magnitude

    # lower bound, upper bound, number of points
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    M0 = 10.**(1.5*float(Mw) + 9.1)
    rho = M0*np.sqrt(2.)

    return UnstructuredGrid({
        'rho': rho*np.ones(N),
        'v': np.zeros(N),
        'w': np.zeros(N),
        'kappa': random(*kappa),
        'sigma': random(*sigma),
        'h': random(*h)},
        callback=tape2015_to_Mij)


def DoubleCoupleGridRegular(moment_magnitude=None, npts_per_axis=25):
    """ Double-couple moment tensor grid with regularly-spaced values
    """ 
    N = npts_per_axis

    if not moment_magnitude:
        raise ValueError
    Mw = moment_magnitude

    # lower bound, upper bound, number of points
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    M0 = 10.**(1.5*float(Mw) + 9.1)
    rho = M0*np.sqrt(2.)

    return Grid({
        'rho': asarray(rho),
        'v': asarray(0.),
        'w': asarray(0.),
        'kappa': regular(*kappa),
        'sigma': regular(*sigma),
        'h': regular(*h)},
        callback=tape2015_to_Mij)


def OriginGrid():
    raise NotImplementedError


def cross():
    """ Cartesian product utility
    """
    raise NotImplementedError



def tape2015_to_Mij(*args, **kwargs):
    """ Converts from Tape2015 parameterization in which the grid is defined
    to Mij parameterization used elsewhere in the code (up-south-east
    convention)
    """
    from mtuq.util.moment_tensor.tape2015 import tt152cmt
    return tt152cmt(*args, **kwargs)

