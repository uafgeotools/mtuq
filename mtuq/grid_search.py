
import numpy as np
import pandas
import xarray

from mtuq.util import iterable, timer, remove_list, warn, ProgressCallback
from mtuq.util.xarray import parse_regular, parse_irregular
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

    try:
        # succeeds only for regularly-spaced grids
        return MTUQDataArray(**parse_regular(origins, sources, values))
    except:
        # fallback for irregularly-spaced grids
        return MTUQDataFrame(**parse_irregular(origins, sources, values))


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

    Almost identical to xarray parent class, except preserves grid search 
    inputs `origins`, `sources` and provides associated methods 
    `best_origin`, `best_source`

    """

    def idxmin(self):
        return self.where(self==self.max(), drop=True).squeeze().coords

    def best_origin(self):
        origins, sources = (self.attrs['origins'], self.attrs['sources'])
        shape = (np.product(sources.shape), len(origins))
        idx = np.unravel_index(np.reshape(self.values, shape).argmin(), shape)[1]
        return origins[idx]

    def best_source(self):
        origins, sources = (self.attrs['origins'], self.attrs['sources'])
        shape = (np.product(sources.shape), len(origins))
        idx = np.unravel_index(np.reshape(self.values, shape).argmin(), shape)[0]
        return sources.get(idx)

    def save(self, filename, *args, **kwargs):
        """ Saves grid search results to NetCDF file
        """
        da = self.copy()
        da.attrs = []
        print('Saving NetCDF file: %s' % filename)
        da.to_netcdf(filename)


class MTUQDataFrame(pandas.DataFrame):
    """ Data structure for storing values on irregularly-spaced grids

    Almost identical to pandas parent class, except preserves grid search  
    inputs `origins`, `sources` and provides associated methods 
    `best_origin`, `best_source`
    """

    def best_origin(self):
        origins, sources = (self.attrs['origins'], self.attrs['sources'])
        shape = (np.product(sources.shape), len(origins))
        idx = np.unravel_index(np.reshape(self.values, shape).argmin(), shape)[1]
        return origins[idx]

    def best_source(self):
        origins, sources = (self.attrs['origins'], self.attrs['sources'])
        shape = (np.product(sources.shape), len(origins))
        idx = np.unravel_index(np.reshape(self.values, shape).argmin(), shape)[0]
        return sources.get(idx)

    def save(self, filename, *args, **kwargs):
        """ Saves grid search results to HDF5 file
        """
        print('Saving HDF5 file: %s' % filename)
        self.to_hdf(filename)


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

