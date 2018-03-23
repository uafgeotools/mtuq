

if __name__=='__main__':
    """
    Double-couple inversion example

    Carries out a grid search over randomly chosen double-couple
    moment tensors, with depth and magnitude varied as well as orientation

    USAGE
       mpirun -n <NPROC> python DCGridSearchMPI.py
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    # FOR NOW THIS IS JUST A STUB

