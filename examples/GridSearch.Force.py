#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_misfit_force
from mtuq.grid import ForceGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.misfit.waveform import calculate_norm_data 



if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double couple moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple.py
    #
    # For a simpler example, see SerialGridSearch.DoubleCouple.py, 
    # which runs the same inversion in serial
    #


    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #

    path_data=    fullpath('data/examples/20210809074550/*[ZRT].sac')
    path_weights= fullpath('data/examples/20210809074550/weights.dat')
    event_id=     '20210809074550'
    model=        'ak135'


    #
    # We are only using surface waves in this example. Check out the DC or FMT examples for multi-mode inversions.
    #

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
    # For our objective function, we will use the L2 norm of the misfit between
    # observed and synthetic waveforms. 
    #

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-35.,
        time_shift_max=+35.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #
    
    grid = ForceGridRegular(magnitudes_in_N=10**np.arange(9,12.1,0.1), npts_per_axis=90)


    # In this example, we use a simple trapezoidal source-time function with a
    # rise time of 4.5 second and a half-duration of 6.75 seconds, obtained from
    # earthquake scaling law.
    wavelet = Trapezoid(
        magnitude=8)


    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
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
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model, include_mt=False, include_force=True)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_sw = None
        greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)

    if comm.rank==0:
        # Computing the norm of the data for the misfit normalization
        norm_sw = calculate_norm_data(data_sw, misfit_sw.norm, ['Z','R','T'])

        results = results_sw/norm_sw

        #
        # Collect information about best-fitting source
        #

        # `grid` index corresponding to minimum misfit force
        idx = results.source_idxmin()

        # Force object
        best_force = grid.get(idx)

        # dictionary of best force direction (F0, phi, h)
        direction_dict = grid.get_dict(idx)

        # dictionary of Fi parameters
        force_dict = best_force.as_dict()

        merged_dict = merge_dicts(
            direction_dict, force_dict, origin)

        #
        # Generate figures and save results
        #

        print('Generating figures...\n')

        plot_data_greens1(event_id+'_force_waveforms.png',
            data_sw, greens_sw, process_sw, misfit_sw, stations, origin, best_force, direction_dict)

        plot_misfit_force(event_id+'_force_misfit.png', results)

        print('Saving results...\n')

        # save best-fitting source
        save_json(event_id+'_force_solution.json', merged_dict)

        # save misfit surface
        results.save(event_id+'_force_misfit.nc')

        print('\nFinished\n')
