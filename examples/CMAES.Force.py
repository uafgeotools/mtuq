#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.stochastic_sampling import initialize_force
from mtuq.stochastic_sampling.cmaes import CMA_ES
from mtuq.graphics import plot_misfit_force
from mtuq.graphics.uq._matplotlib import _plot_force_matplotlib



if __name__=='__main__':
    #
    # Carries out CMA-ES inversion over moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python CMAES.Force.py
    # ---------------------------------------------------------------------
    # The code is intended to be run in parallel, although the `greens` mode
    # exhibits some scaling issues (potentially due to IO and MPI comm overheads).
    # ---------------------------------------------------------------------
    # The `greens` mode with 24 ~ 120 mutants per generation (CMAES parameter 'lambda')
    # should only take a few minutes to run on a single core, and achieves better 
    # results than when using a grid search. (No restriction of being on a grid, including 
    # finer Mw search).
    # The algorithm can converge with as low as 6 mutants per generation, but this is
    # not recommended as it will take a longer time to converge, and is more prone to
    # getting stuck in local minima. This could be useful if you are trying to find
    # other minima, but is not recommended for general use.
    # ---------------------------------------------------------------------
    # The 'database' should be used when searching over depth / hypocenter.
    # I recommend anything between 24 to 120 mutants per generation, (CMAES parameter 'lambda')
    # Each mutant will require its own greens functions, meaning the most compute time will be
    # spent fetching and pre-processing greens functions. This can be sped up by using a
    # larger number of cores, but the scaling is not perfect. (e.g. 24 cores is not 24x faster)
    # ---------------------------------------------------------------------
    # CMA-ES algorithm
    # 1 - Initialise the CMA-ES algorithm with a set of mutants
    # 2 - Evaluate the misfit of each mutant
    # 3 - Sort the mutants by misfit (best to worst), the best mutants are used to update the
    #     mean and covariance matrix of the next generation (50% of the population retained)
    # 4 - Update the mean and covariance matrix of the next generation
    # 5 - Repeat steps 2-4 until the ensemble of mutants converges

    path_data=    fullpath('data/examples/20210809074550/*[ZRT].sac')
    path_weights= fullpath('data/examples/20210809074550/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'
    mode =        'greens' # 'database' or 'greens'

    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-35.,
        time_shift_max=+35.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function. Note that these should be functional in the CMAES
    # mode.
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the source wavelet. 
    #

    wavelet = Trapezoid(
        magnitude=8)


    #
    # The Origin time and hypocenter are defined as in the grid-search codes
    # It will either be fixed and used as-is by the CMA-ES mode (typically `greens` mode)
    # or will be used as a starting point for hypocenter search (using the `database` mode)
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2021-08-09T07:45:50.000000Z',
        'latitude': 61.24,
        'longitude': -147.96,
        'depth_in_m': 0,
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()


        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        if mode == 'greens':
            print('Reading Greens functions...\n')
            greens = download_greens_tensors(stations, origin, model, include_mt=False, include_force=True)
            # ------------------
            # Alternatively, if you have a local AxiSEM database, you can use:
            # db = open_db('/Path/to/Axisem/Database/ak135f/', format='AxiSEM', include_mt=False, include_force=True)
            # greens = db.get_greens_tensors(stations, origin, model)
            # ------------------
            greens.convolve(wavelet)
            greens_bw = greens.map(process_bw)
            greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_bw = None
        data_sw = None
        if mode == 'greens':
            db = None
            greens_bw = None
            greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    if mode == 'greens':
        greens_bw = comm.bcast(greens_bw, root=0)
        greens_sw = comm.bcast(greens_sw, root=0)
    elif mode == 'database':
        # This mode expects the path to a local AxiSEM database to be specified 
        db = open_db('/Path/to/Axisem/Database/ak135f/', format='AxiSEM', include_mt=False, include_force=True)

    #
    # The main computational work starts now
    #
    if mode == 'database':
        parameter_list = initialize_force(F0_range=[1e10, 1e12], depth=[0, 1000])
    elif mode == 'greens':
        parameter_list = initialize_force(F0_range=[1e10, 1e12])

    DATA = [data_sw]  # add more as needed
    MISFIT = [misfit_sw]  # add more as needed
    PROCESS = [process_sw]  # add more as needed
    GREENS = [greens_sw] if mode == 'greens' else None  # add more as needed

    popsize = 48 # -- CMA-ES population size (you can play with this value)
    CMA = CMA_ES(parameter_list , origin=origin, lmbda=popsize, event_id=event_id)
    CMA.sigma = 2 # -- CMA-ES step size, defined as 1 standard deviation of the initial parameter distribution (you can play with this value, higher values are best for exploration and are generaly worth it)
    iter = 120 # -- Number of iterations (you can play with this value)

    if mode == 'database':
        CMA.Solve(DATA, stations, MISFIT, PROCESS, db, iter, wavelet, plot_interval=10, misfit_weights=[1.])
    elif mode == 'greens':
        CMA.Solve(DATA, stations, MISFIT, PROCESS, GREENS, iter, plot_interval=10, misfit_weights=[1.])

    #  --- Plotting force misfit result
    # CMA.mutants_logger_list returns an object similar to mtuq defautl grid search result, which is compatible with all plotting functions
    result = CMA.mutants_logger_list # -- This is the list of mutants (i.e. the population) at each iteration

    # Plotting the result - plot_type can be 'colormesh' or 'contour', but in the case of CMA-ES, colormesh gives better result than a contour plot 
    plot_misfit_force(event_id+'_misfit_force.png', result, backend=_plot_force_matplotlib, plot_type='colormesh')

    if comm.rank==0:
        print('\nFinished\n')