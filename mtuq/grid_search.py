
import numpy as np
import pandas
import xarray

from collections.abc import Iterable
from mtuq.event import Origin
from mtuq.grid import DataFrame, DataArray, Grid, UnstructuredGrid
from mtuq.util import gather2, iterable, timer, remove_list, warn,\
    ProgressCallback, dataarray_idxmin, dataarray_idxmax
from os.path import splitext
from xarray.core.formatting import unindexed_dims_repr


xarray.set_options(keep_attrs=True)


def grid_search(data, greens, misfit, origins, sources, 
    msg_interval=25, timed=True, verbose=1, gather=True):

    """ Evaluates misfit over grids

    .. rubric :: Usage

    Carries out a grid search by evaluating 
    `misfit(data, greens.select(origin), source)` over all origins and sources.

    If `origins` and `sources` are regularly-spaced, returns an `MTUQDataArray`
    containing misfit values and corresponding grid points. Otherwise, 
    an `MTUQDataFrame` is returned.


    .. rubric :: Input arguments


    ``data`` (`mtuq.Dataset`):
    The observed data to be compared with synthetic data


    ``greens`` (`mtuq.GreensTensorList`):
    Green's functions used to generate synthetic data


    ``misfit`` (`mtuq.Misfit` or some other function):
    Misfit function


    ``origins`` (`list` of `mtuq.Origin` objects):
    Origins to be searched over


    ``sources`` (`mtuq.Grid` or `mtuq.UnstructuredGrid`):
    Source mechanisms to be searched over


    ``msg_interval`` (`int`):
    How frequently, as a percentage of total evaluations, should progress 
    messages be displayed? (value between 0 and 100)


    ``timed`` (`bool`):
    Displays elapsed time at end


    ``gather`` (`bool`):
    If `True`, process 0 returns all results and any other processes return
    `None`.  Otherwise, results are divided evenly among processes.
    (ignored outside MPI environment)


    .. note:

      If invoked from an MPI environment, the grid is partitioned between
      processes and each process runs ``_grid_search_serial`` on its given
      partition. If not invoked from an MPI environment, `grid_search`
      reduces to ``_grid_search_serial``.

    """
    origins = iterable(origins)
    for origin in origins:
        assert type(origin) is Origin

    if type(sources) not in (Grid, UnstructuredGrid):
        raise TypeError

    _subset = None
    if _is_mpi_env():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        iproc, nproc = comm.rank, comm.size

        if nproc > sources.size:
            raise Exception('Number of CPU cores exceeds size of grid')

        # partition grid and scatter across processes
        _subsets = None
        if iproc == 0:
            _subsets = sources.partition(nproc)
        _subset = comm.scatter(_subsets, root=0)
        if iproc != 0:
            timed = False
            msg_interval = 0


    # print number of grid points
    if verbose>0 and _is_mpi_env() and iproc==0:

        print('  Number of grid points: %.3e' %\
            (len(origins)*len(sources)))

        print('  Number of MPI processes: %d\n' % nproc)

    elif verbose>0 and not _is_mpi_env():

        print('  Number of misfit evaluations: %.3e\n' %\
            (len(origins)*len(sources)))


    # evaluate misfit over origins and sources
    values = _grid_search_serial(
        data, greens, misfit, origins, _subset or sources, timed=timed,
        msg_interval=msg_interval)


    if _is_mpi_env() and gather:
        if iproc==0:
            values = gather2(comm, array)
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

    .. note::

        Besides the methods below, `MTUQDataArray` includes many useful methods
        inherited from ``xarray.DataArray``. See 
        `xarray documentation <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_ 
        for more information.

    """

    def idxmin(self, idx_type=None):
        """ Returns coordinates corresponding to minimum misfit
        """
        if idx_type is None:
            return dataarray_idxmin(self)

        elif idx_type in ('origin', 'origin_idx'):
            return int(dataarray_idxmin(self)['origin_idx'])

        elif idx_type in ('source', 'source_idx'):
            shape = self._get_shape()
            return np.unravel_index(self.argmin(), shape)[0]

        else:
            raise TypeError

    def idxmax(self, idx_type=None):
        """ Returns coordinates corresponding to maximum misfit
        """
        if idx_type is None:
            return dataarray_idxmax(self)

        elif idx_type in ('origin', 'origin_idx'):
            return int(dataarray_idxmax(self)['origin_idx'])

        elif idx_type in ('source', 'source_idx'):
            shape = self._get_shape()
            return np.unravel_index(self.argmax(), shape)[0]

        else:
            raise TypeError

    def _get_shape(self):
        """ Private helper method
        """
        nn = len(self.coords['origin_idx'])
        return (int(self.size/nn), nn)

    def save(self, filename, *args, **kwargs):
        """ Saves grid search results to NetCDF file
        """
        print('  saving NetCDF file: %s' % filename)
        self.to_netcdf(filename)

    def __repr__(self):
        summary = [
            'Summary:',
            '  grid shape: %s' % self.shape.__repr__(),
            '  grid size:  %d' % self.size,
            '  mean: %.3e' % np.mean(self.values),
            '  std:  %.3e' % np.std(self.values),
            '  min:  %.3e' % self.values.min(),
            '  max:  %.3e' % self.values.max(),
            '',
        ]

        if hasattr(self, "coords"):
            if self.coords:
                summary.append(repr(self.coords))

            unindexed_dims_str = unindexed_dims_repr(self.dims, self.coords)
            if unindexed_dims_str:
                summary.append(unindexed_dims_str)

        return "\n".join(summary+[''])



class MTUQDataFrame(pandas.DataFrame):
    """ Data structure for storing values on irregularly-spaced grids

    .. note::

        Besides the methods below, `MTUQDataFrame` includes many useful methods
        inherited from ``pandas.DataFrame``. See 
        `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        for more information.

    """
    def idxmin(self, idx_type=None):
        """ Returns coordinates corresponding to minimum misfit
        """
        if idx_type is None:
            return self[0].idxmin()

        elif idx_type in ('origin', 'origin_idx'):
            df = self.reset_index()
            return df['origin_idx'][df[0].idxmin()]

        elif idx_type in ('source', 'source_idx'):
            df = self.reset_index()
            return df['source_idx'][df[0].idxmin()]

        else:
            raise TypeError

    def idxmax(self, idx_type=None):
        """ Returns coordinates corresponding to maximum misfit
        """
        if idx_type is None:
            return self[0].idxmax()

        elif idx_type in ('origin', 'origin_idx'):
            df = self.reset_index()
            return df['origin_idx'][df[0].idxmax()]

        elif idx_type in ('source', 'source_idx'):
            df = self.reset_index()
            return df['source_idx'][df[0].idxmax()]

        else:
            raise TypeError

    def save(self, filename, *args, **kwargs):
        """ Saves grid search results to HDF5 file
        """
        print('  saving HDF5 file: %s' % filename)
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
    origin_dims = ('origin_idx',)
    origin_coords = [np.arange(len(origins))]
    origin_shape = (len(origins),)

    source_dims = sources.dims
    source_coords = sources.coords
    source_shape = sources.shape

    return MTUQDataArray(**{
        'data': np.reshape(values, source_shape + origin_shape),
        'coords': source_coords + origin_coords,
        'dims': source_dims + origin_dims,
         })


def _to_dataframe(origins, sources, values, index_type=2):
    """ Converts grid_search inputs to DataFrame
    """
    if len(origins)*len(sources) > 1.e7:
        print("  pandas indexing becomes very slow with >10 million rows\n"
              "  consider using index_type=1 in mtuq.grid_search._to_dataframe\n"
             )

    origin_idx = np.arange(len(origins), dtype='int')
    source_idx = np.arange(len(sources), dtype='int')

    # Cartesian products
    origin_idx = list(np.repeat(origin_idx, len(sources)))
    source_idx = list(np.tile(source_idx, len(origins)))
    source_coords = []
    for _i, coords in enumerate(sources.coords):
        source_coords += [list(np.tile(coords, len(origins)))]

    # assemble coordinates
    coords = [origin_idx, source_idx]
    dims = ('origin_idx', 'source_idx')
    if index_type==2:
        coords += source_coords
        dims += sources.dims

    # construct DataFrame
    data = {dims[_i]: coords[_i] for _i in range(len(dims))}
    data.update({0: values.flatten()})
    df = MTUQDataFrame(data=data)
    df = df.set_index(list(dims))
    return df


