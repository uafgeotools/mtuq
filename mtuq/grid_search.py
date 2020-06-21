
import numpy as np
import pandas
import xarray

from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import iterable, timer, remove_list, warn, ProgressCallback
from os.path import splitext

xarray.set_options(keep_attrs=True)



def grid_search(data, greens, misfit, origins, sources, 
    msg_interval=25, timed=True):

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

    if _is_mpi_env():
        # partition grid and scatter across processes
        if iproc == 0:
            sources = sources.partition(nproc)
        sources = comm.scatter(sources, root=0)

        if iproc != 0:
            timed = False
            msg_interval = 0

    values = _grid_search_serial(
        data, greens, misfit, origins, sources, timed=timed, 
        msg_interval=msg_interval)

    if _is_mpi_env():
        values = np.concatenate(comm.allgather(values))

    if issubclass(type(sources), Grid):
        return MTUQDataArray(**_parse_regular(origins, sources, values))

    elif issubclass(type(sources), UnstructuredGrid):
        return MTUQDataFrame(**_parse_irregular(origins, sources, values))

    else:
        raise TypeError


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
        """ Returns coordinates dictionary corresponding to minimum misfit
        """
        # eventually this will be implemented directly in xarray.DataFrame
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


def _parse_regular(origins, sources, values):
    """ Converts grid_search inputs to DataArray inputs
    """
    from mtuq.grid import Grid

    origin_dims = ('origin_idx',)
    origin_coords = [np.arange(len(origins))]
    origin_shape = (len(origins),)

    source_dims = sources.dims
    source_coords = sources.coords
    source_shape = sources.shape

    attrs = {
        'origins': origins,
        'sources': sources,
        'origin_dims': origin_dims,
        'origin_coords': origin_coords,
        'origin_shape': origin_shape,
        'source_dims': source_dims,
        'source_coords': source_coords,
        'source_shape': source_shape,
        }

    return {
        'data': np.reshape(values, source_shape + origin_shape),
        'coords': source_coords + origin_coords,
        'dims': source_dims + origin_dims,
        #'attrs': attrs,
        }


def _parse_irregular(origins, sources, values):
    """ Converts grid_search inputs to DataFrame inputs
    """
    if not issubclass(type(sources), UnstructuredGrid):
        raise TypeError

    origin_idx = np.arange(len(origins), dtype='int')
    origin_idx = list(np.repeat(origin_idx, len(sources)))

    source_idx = np.arange(len(sources.coords[0]), dtype='int')
    source_idx = list(np.tile(source_idx, len(origins)))

    source_coords = []
    for _i, coords in enumerate(sources.coords):
        source_coords += [list(np.tile(coords, len(origins)))]

    coords = [origin_idx, source_idx] + source_coords
    dims = ('origin_idx', 'source_idx') + sources.dims

    return {
        'data': {'misfit': values.flatten()},
        'index': pandas.MultiIndex.from_tuples(zip(*coords), names=dims),
        }


def _split(dims=('latitude', 'longitude', 'depth_in_m')):
    """ Tries to split origin coordinates into multidimensional array 
    (fails if origins aren't reguarly spaced)
    """

    ni = len(origins)
    nj = len(origin_dims)

    array = np.empty((ni, nj))
    for _i, origin in enumerate(origins):
        for _j, dim in enumerate(origin_dims):
            array[_i,_j] = origin[dim]

    coords_uniq = []
    shape = ()
    for _j, dim in enumerate(origin_dims):
        coords_uniq += [np.unique(array[:,_j])]
        shape += (len(coords_uniq[-1]),)

    if np.product(shape)==ni:
        origin_coords, origin_shape = coords_uniq, shape
    else:
        raise TypeError

