

import numpy as np

from builtins import object
from numpy import pi as PI
from numpy.random import uniform as random
from mtuq.util import AttribDict, asarray
from mtuq.util.math import open_interval as regular
#from mtuq.util.moment_tensor.TapeTape2015 import to_mij
from mtuq.util.lune import to_mij, spherical_to_Cartesian




class Grid(object):
    """
    A regularly-spaced grid defined by values along axes

    .. rubric:: Examples

    To cover the unit square with an `N`-by-`N` rectangular grid:

    .. code ::

       grid = Grid(('x': np.linspace(0., 1., N)),
                   ('y': np.linpsace(0., 1., N)))


    To parameterize the surface of the Earth with an `N`-by-`2N` Mercator grid:

    .. code::          

       grid = Grid(('latitude': np.linspace(-90., 90., N)),
                   ('longitude': np.linspace(-180., 180., 2*N)))


    .. rubric:: Iterating on grids

    The order of iteration over a grid is determined by the order of keys in
    the dictionary input argument. In the unit square example above, ``'x'`` is
    the fast axis and ``'y'`` is the slow axis.

    """
    def __init__(self, items, start=0, stop=None, callback=None):
        self.items = items

        # list of parameter names
        self.keys = [item[0] for item in items]
        
        # corresponding list of axis arrays
        self.vals = [item[1] for item in items]

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

 
    def as_array(self):
        """ Returns `i-th` point of grid
        """
        array = np.zeros((self.size, self.ndim))
        for _i in range(self.size):
            array[_i, :] = self.get(_i)
        return array


    def get(self, i):
        """ Returns `i-th` point of grid
        """
        vals = self.vals
        array = np.zeros(self.ndim)

        for _k in range(self.ndim):
            val = vals[_k]
            array[_k] = val[int(i%len(val))]
            i/=len(val)

        if self.callback:
            return self.callback(*array)
        else:
            return array


    def partition(self, nproc):
        """ Partitions grid for parallel processing
        """
        if self.start!=0:
            raise Exception

        subsets = []
        for iproc in range(nproc):
            start=int(iproc*self.size/nproc)
            stop=(iproc+1)*int(self.size/nproc)
            subsets += [Grid(self.items, start, stop, callback=self.callback)]
        return subsets


    def save(self, filename, items={}):
        """ Saves a set of values defined on grid
        """
        import h5py
        with h5py.File(filename, 'w') as hf:
            for key, val in zip(self.keys, self.vals):
                hf.create_dataset(key, data=val)

            for key, val in items.iteritems():
                hf.create_dataset(key, data=val)


    def __len__(self):
        return self.size



    # the next two methods make it possible to iterate over the grid
    def __next__(self): 
        """ Advances iteration index
        """
        if self.index < self.stop:
           # get the i-th point in grid
           p = self.get(self.index)
        else:
            self.index = self.start
            raise StopIteration
        self.index += 1
        return p


    def __iter__(self):
        return self



class UnstructuredGrid(object):
    """ 
    A grid defined by a list of individual coordinate points, which can be
    irregularly spaced

    .. rubric:: Example

    Unstructured grid consisting of `N` randomly-chosen points within the unit 
    square:

    .. code ::

      grid = UnstructuredGrid((('x': np.random.rand(N)),
                               ('y': np.random.rand(N)))

    """
    def __init__(self, items, start=0, stop=None, callback=None):
        self.items = items

        # list of parameter names
        self.keys = [item[0] for item in items]

        # corresponding list of axis arrays
        self.vals = [item[1] for item in items]

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


    def as_array(self):
        """ Returns `i-th` point of grid
        """
        array = np.zeros((self.size, self.ndim))
        for _i in range(self.size):
            array[_i, :] = self.get(_i+self.start)
        return array


    def get(self, i):
        """ Returns `i-th` point of grid
        """
        i -= self.start
        vals = self.vals
        array = np.zeros(self.ndim)

        for _k in range(self.ndim):
            array[_k] = vals[_k][i]

        if self.callback:
            return self.callback(*array)
        else:
            return array


    def partition(self, nproc):
        """ Partitions grid for parallel processing
        """
        subsets = []
        for iproc in range(nproc):
            start=iproc*int(self.size/nproc)
            stop=(iproc+1)*int(self.size/nproc)
            items = []
            for key, val in self.items:
                items += [[key, val[start:stop]]]
            subsets += [UnstructuredGrid(items, start, stop, callback=self.callback)]
        return subsets


    def save(self, filename, items={}):
        """ Saves a set of values defined on grid
        """
        import h5py
        with h5py.File(filename, 'w') as hf:
            for key, val in zip(self.keys, self.vals):
                hf.create_dataset(key, data=val)

            for key, val in items.iteritems():
                hf.create_dataset(key, data=val)


    def __len__(self):
        return self.size


    # the next two methods make it possible to iterate over the grid
    def __next__(self): 
        """ Advances iteration index
        """
        if self.index < self.stop:
           # get the i-th point in grid
           p = self.get(self.index)
        else:
            self.index = self.start
            raise StopIteration
        self.index += 1
        return p


    def __iter__(self):
        return self


def FullMomentTensorGridRandom(magnitude=None, npts=1000000, callback=to_mij):
    """ Full moment tensor grid with randomly-spaced values
    """
    magnitude, count = _check_magnitude(magnitude)
    N = npts*count

    # lower bound, upper bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = np.zeros((count, npts))
    for _i, Mw in enumerate(magnitude):
        M0 = 10.**(1.5*float(Mw) + 9.1)
        rho[_i, :] = M0*np.sqrt(2.)

    return UnstructuredGrid((
        ('rho', rho.flatten()),
        ('v', random(*v)),
        ('w', random(*w)),
        ('kappa', random(*kappa)),
        ('sigma', random(*sigma)),
        ('h', random(*h))),
        callback=callback)


def FullMomentTensorGridRegular(magnitude=None, npts_per_axis=20, callback=to_mij):
    """ Full moment tensor grid with regularly-spaced values
    """
    magnitude, count = _check_magnitude(magnitude)
    N = npts_per_axis

    # lower bound, upper bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = []
    for Mw in magnitude:
        M0 = 10.**(1.5*float(Mw) + 9.1)
        rho += [M0*np.sqrt(2)]

    return Grid((
        ('rho', asarray(rho)),
        ('v', regular(*v)),
        ('w', regular(*w)),
        ('kappa', regular(*kappa)),
        ('sigma', regular(*sigma)),
        ('h', regular(*h))),
        callback=callback)


def DoubleCoupleGridRandom(magnitude=None, npts=50000, callback=to_mij):
    """ Double-couple moment tensor grid with randomly-spaced values
    """
    magnitude, count = _check_magnitude(magnitude)
    N = npts*count

    # lower bound, upper bound, number of points
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = np.zeros((count, npts))
    for _i, Mw in enumerate(magnitude):
        M0 = 10.**(1.5*float(Mw) + 9.1)
        rho[_i, :] = M0*np.sqrt(2.)

    return UnstructuredGrid((
        ('rho', rho.flatten()),
        ('v', np.zeros(N)),
        ('w', np.zeros(N)),
        ('kappa', random(*kappa)),
        ('sigma', random(*sigma)),
        ('h', random(*h))),
        callback=callback)


def DoubleCoupleGridRegular(magnitude=None, npts_per_axis=40, callback=to_mij):
    """ Double-couple moment tensor grid with regularly-spaced values
    """ 
    magnitude, count = _check_magnitude(magnitude)
    N = npts_per_axis

    # lower bound, upper bound, number of points
    kappa = [0., 360, N]
    sigma = [-180., 180., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = []
    for Mw in magnitude:
        M0 = 10.**(1.5*float(Mw) + 9.1)
        rho += [M0*np.sqrt(2)]

    return Grid((
        ('rho', asarray(rho)),
        ('v', asarray(0.)),
        ('w', asarray(0.)),
        ('kappa', regular(*kappa)),
        ('sigma', regular(*sigma)),
        ('h', regular(*h))),
        callback=callback)



def ForceGridRegular(magnitude=None, npts_per_axis=40):
    """ Full moment tensor grid with randomly-spaced values
    """
    raise NotImplementedError


def ForceGridRandom(magnitude=None, npts=50000, callback=spherical_to_Cartesian):
    """ Full moment tensor grid with randomly-spaced values
    """
    magnitude, count = _check_magnitude(magnitude)
    N = npts*count

    theta = [0., 180, N]
    phi = [0., 360., N]

    # magnitude is treated separately
    r = np.zeros((count, npts))
    for _i, Mw in enumerate(magnitude):
        M0 = 10.**(1.5*float(Mw) + 9.1)
        r[_i, :] = M0*np.sqrt(2.)

    return UnstructuredGrid({
        'r': r.flatten(),
        'theta': random(*theta),
        'phi': random(*phi)},
        callback=callback)


def _check_magnitude(M):
    if type(M) in [np.ndarray, list, tuple]:
        count = len(M)
    elif type(M) in [int, float]:
        M = [float(M)]
        count = 1
    else:
        raise TypeError
    return M, count

