
import numpy as np
import pandas
import xarray

from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import iterable, timer, remove_list, warn, ProgressCallback
from os.path import splitext

xarray.set_options(keep_attrs=True)



def grid_search(data, greens, misfit, origins, sources, 
    msg_interval=25, timed=True, gather=True):

    """ Evaluates misfit over grids

    .. rubric :: Usage

    Carries out a grid search by evaluating 
    `misfit(data, greens.select(origin), source)` over all origins and sources.

    If `origins` and `sources` are regularly-spaced, returns an `MTUQDataArray`
    containing misfit values and corresponding grid points; otherwise, returns
    an `MTUQDataFrame`.


    .. rubric :: Input arguments


    ``data`` (`mtuq.dataset.Dataset`):
    The observed data that will be compared with synthetic data


    ``greens`` (`mtuq.greens_tensor.GreensTensorList`):
    Green's functions that will be used to generate synthetic data


    ``misfit`` (`mtuq.misfit.Misfit` or some other function):
    Implements norm of data and synthetics


    ``origins`` (`list` of `mtuq.source.Origin` objects)
    Origins that will be used to generate synthetics


    ``sources`` (`mtuq.grid.Grid` or `mtuq.grid.UnstructuredGrid`):
    Source mechanisms that will be used to generate synthetics


    ``msg_interval`` (`int`):
    How frequently, as a percentage of total evaluations, should progress 
    messages be displayed? (value between 0 and 100)


    ``timed`` (`bool`):
    Display elapsed time at end?


    ``gather`` (`bool`):
    If True, process 0 returns all results and any other processes return None.
    If False, results are divided evenly among processes. 
    (Ignored outside MPI environment.)


    .. note:

      If invoked from an MPI environment, the grid is partitioned between
      processes and each process runs ``_grid_search_serial`` on its given
      partition. If not invoked from an MPI environment, `grid_search`
      reduces to ``_grid_search_serial``.

    """
    if _is_mpi_env():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        iproc, nproc = comm.rank, comm.size
        if nproc > sources.size:
            raise Exception('Number of CPU cores exceeds size of grid')


    origins = iterable(origins)
    sources = iterable(sources)
    subsets = None
    subset = None

    if _is_mpi_env():
        # partition grid and scatter across processes
        if iproc == 0:
            subsets = sources.partition(nproc)
        subset = comm.scatter(subsets, root=0)
        if iproc != 0:
            timed = False
            msg_interval = 0


    # evaluate misfit over origins and sources
    values = _grid_search_serial(
        data, greens, misfit, origins, subset or sources, timed=timed, 
        msg_interval=msg_interval)

    if _is_mpi_env() and gather:
        values = comm.gather(values, root=0)
        if iproc == 0:
            values = np.concatenate(values, axis=0)
        else:
            return


    # convert from NumPy array to DataArray or DataFrame
    if issubclass(type(sources), Grid):
        return _to_dataarray(origins, sources, values)

    elif issubclass(type(sources), UnstructuredGrid):
        return _to_dataframe(origins, sources, values)


@timer
def _grid_search_serial(data, greens, misfit, origins, sources, 
    timed=True, msg_interval=25):
    """ Evaluates misfit over origin and source grids 
    (serial implementation)
    """
    ni = len(origins)
    nj = len(sources)

    values = []
    for _i, origin in enumerate(origins):

        msg_handle = ProgressCallback(
            start=_i*nj, stop=ni*nj, percent=msg_interval)

        # evaluate misfit function
        values += [misfit(
            data, greens.select(origin), sources, msg_handle)]

    # returns NumPy array of shape `(len(sources), len(origins))` 
    return np.concatenate(values, axis=1)



class MTUQDataArray(xarray.DataArray):
    """ Data structure for storing values on regularly-spaced grids
    """

    def idxmin(self):
        """ Returns coordinates corresponding to minimum misfit
        """
        # idxmin has been implemented in a beta version of xarray; this
        # subclass method could be removed if idxmin every becomes part of the
        # main xarray package
        return self.where(self==self.max(), drop=True).squeeze().coords

    def origin_idx(self):
        """ Returns origin index corresponding to minimum misfit
        """
        return int(self.idxmin()['origin_idx'])

    def source_idx(self):
        """ Returns source index corresponding to minimum misfit
        """
        shape = self._get_shape()
        return np.unravel_index(self.argmin(), shape)[0]

    def _get_shape(self):
        """ Private helper method
        """
        nn = len(self.coords['origin_idx'])
        return (int(self.size/nn), nn)

    def save(self, filename, *args, **kwargs):
        """ Saves grid search results to NetCDF file
        """
        print('Saving NetCDF file: %s' % filename)
        self.to_netcdf(filename)



class MTUQDataFrame(pandas.DataFrame):
    """ Data structure for storing values on irregularly-spaced grids
    """

    def origin_idx(self):
        """ Returns origin index corresponding to minimum misfit
        """
        df = self.reset_index()
        return df.idxmin()['origin_idx']

    def source_idx(self):
        """ Returns source index corresponding to minimum misfit
        """
        df = self.reset_index()
        return df.idxmin()['source_idx']

    def save(self, filename, *args, **kwargs):
        """ Saves grid search results to HDF5 file
        """
        print('Saving HDF5 file: %s' % filename)
        df = pandas.DataFrame(self.values, index=self.index)
        df.to_hdf(filename, key='df', mode='w')

    @property
    def _constructor(self):
        return MTUQDataFrame


#
# utility functions
#

def _is_mpi_env():
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.Get_size()>1:
        return True
    else:
        return False


def _to_dataarray(origins, sources, values):
    """ Converts grid_search inputs to DataArray
    """
    from mtuq.grid import Grid

    origin_dims = ('origin_idx',)
    origin_coords = [np.arange(len(origins))]
    origin_shape = (len(origins),)

    source_dims = sources.dims
    source_coords = sources.coords
    source_shape = sources.shape

    kwargs = {
        'data': np.reshape(values, source_shape + origin_shape),
        'coords': source_coords + origin_coords,
        'dims': source_dims + origin_dims,
         }

    return MTUQDataArray(**kwargs)


def _to_dataframe(origins, sources, values, index_type='columns'):
    """ Converts grid_search inputs to DataFrame
    """
    if not issubclass(type(sources), UnstructuredGrid):
        raise TypeError

    if len(origins)*len(sources) > 1.e6:
        print("  Working with large DataFrames can a very long time")

    origin_idx = np.arange(len(origins), dtype='int')
    source_idx = np.arange(len(sources), dtype='int')

    # Cartesian products
    origin_idx = list(np.repeat(origin_idx, len(sources)))
    source_idx = list(np.tile(source_idx, len(origins)))
    source_coords = []
    for _i, coords in enumerate(sources.coords):
        source_coords += [list(np.tile(coords, len(origins)))]


    # idea 1 - much too slow!
    #coords = [origin_idx, source_idx]
    #dims = ('origin_idx', 'source_idx')
    #coords += source_coords
    #dims += sources.dims
    #data = {'values': values.flatten()}
    #index = pandas.MultiIndex.from_tuples(zip(*coords), names=dims)
    #return MTUQDataFrame(data=data, index=index)


    # idea 2 - ugly but fast!
    coords = [origin_idx, source_idx]
    dims = ('origin_idx', 'source_idx')
    coords += source_coords
    dims += sources.dims
    data = {dims[_i]: coords[_i] for _i in range(len(dims))}
    data.update({'values': values.flatten()})
    df = MTUQDataFrame(data=data)
    return df.set_index(list(dims))



