
import numpy as np
from mtuq.util import iterable, timer, warn, ProgressCallback



def grid_search(data, greens, misfit, origins, sources, 
    msg_interval=25, timed=True):

    """ Evaluates misfit over grids

    .. rubric :: Usage

    Carries out a grid search by evaluating 
    `misfit(data, greens.select(origin), source)` for all origins and sources.
    Returns a NumPy array of misfit values of shape 
    `(len(sources), len(origins))` 

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
            raise Exception('Number of cores exceeds size of grid')

        # partition grid and scatter across processes
        if iproc == 0:
            sources = sources.partition(nproc)
        sources = comm.scatter(sources, root=0)

        if iproc != 0:
            timed = False
            msg_interval = 0

    # NumPy array of misfit values of shape `(len(sources), len(origins))` 
    values = _grid_search_serial(
        data, greens, misfit, origins, sources, timed=timed, 
        msg_interval=msg_interval)

    if _is_mpi_env():
        values = np.concatenate(comm.allgather(values))

    #if type(sources)==Grid:
    #    return MTUQDataArray(sources, origins, values)
    #
    #elif type(sources)==UnstructuredGrid:
    #    return MTUQDataFrame(sources, origins, values)

    return values



@timer
def _grid_search_serial(data, greens, misfit, origins, sources, 
    timed=True, msg_interval=25):
    """ Evaluates misfit over origin and source grids 
    (serial implementation)
    """
    origins = iterable(origins)
    ni = len(origins)
    nj = len(sources)

    values = []
    for _i, origin in enumerate(origins):

        msg_handle = ProgressCallback(
            start=_i*nj, stop=ni*nj, percent=msg_interval)

        # evaluate misfit function
        values += [misfit(
            data, greens.select(origin), sources, msg_handle)]

    return np.concatenate(values, axis=1)



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


def MTUQDataArray(DataArray):
    def __init__(self, sources, origins, values):
        raise NotImplementedError


    def best_source(self):
        raise NotImplementedError


    def best_origin(self):
        raise NotImplementedError



def MTUQDataFrame(DataFrame):
    def __init__(self, sources, origins, values):
        raise NotImplementedError


    def best_source(self):
        raise NotImplementedError


    def best_origin(self):
        raise NotImplementedError

