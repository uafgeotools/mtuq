
import warnings
import numpy as np

from mtuq.util.util import AttribDict, warn

try:
    import h5py
except:
    warn('Could not import h5py.')



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



