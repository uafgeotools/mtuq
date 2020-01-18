
import numpy as np
from mtuq.util import iterable, timer



def grid_search(data, greens, misfit, origins, sources, 
    allgather=True, verbose=True):

    """ Evaluates misfit over grids

    .. rubric :: Usage

    Carries out a grid search by evaluating `misfit(data, greens)` over
    origin and source grids.  Returns an array of misfit values of shape 
    ``(len(sources), len(origins))`` 

    .. note:

      If invoked from an MPI environment, the grid is partitioned between
      processes and each process runs ``grid_search_serial`` on its given
      partition. If not invoked from an MPI environment, this function simply 
      reduces to ``grid_search_serial``.


    .. rubric :: Parameters

    ``data``
    Must be ``mtuq.dataset.Dataset``

    ``greens``
    Must be ``mtuq.greens_tensor.GreensTensorList``.

    ``misfit``
    Can be ``mtuq.misfit.Misfit`` or some other function that acts on the first
    two arguments

    ``origins``
    Must be a `list` of ``mtuq.source.Origin`` objects

    ``sources``
    Must be ``mtuq.grid.Grid``or ``mtuq.grid.UnstructuredGrid``

    ``allgather``
    Whether or not to combine results from all processes, if invoked from an
    MPI environment
            
    """
    if _is_mpi_env():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        iproc, nproc = comm.rank, comm.size

        # partition grid and scatter across processes
        if iproc == 0:
            sources = sources.partition(nproc)
        sources = comm.scatter(sources, root=0)

        if iproc != 0:
            verbose = False

    results = grid_search_serial(
        data, greens, misfit, origins, sources, verbose=verbose)

    if allgather and _is_mpi_env():
        # all processes share results
        return np.concatenate(comm.allgather(results))
    else:
        # each process just returns its own results
        return results



@timer
def grid_search_serial(data, greens, misfit, origins, sources, 
    verbose=True):
    """ Evaluates misfit over origin and source grids 
    (serial implementation)
    """
    results = []
    for origin in iterable(origins):
        results += [misfit(data, greens.select(origin), sources, verbose)]
    return np.concatenate(results, axis=1)



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


