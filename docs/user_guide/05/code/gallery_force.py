#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.grid import ForceGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid

from mtuq.graphics import plot_misfit_force, plot_likelihood_force, plot_magnitude_tradeoffs_force
from mtbench import calculate_sigma



if __name__=='__main__':
    #
    # Reproduces figures shown in mtuq/docs/user_guide/gallery_force.rst
    #
    # USAGE
    #   mpirun -n <NPROC> python gallery_force.py
    #   

    #
    # WARNING
    #
    # This script is likely to break at some point because it is not 
    # automatically generated or continuously tested
    #
    # For a more robust but otherwise very similar example, see
    # mtuq/examples/GridSearchFullMomentTensor.py
    #

    
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'


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


    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )


    station_id_list = parse_station_codes(path_weights)


    grid = ForceGridRegular(
        npts_per_axis=25,
        magnitudes_in_N=10.**np.arange(11.,12.,0.005))


    wavelet = Trapezoid(
        magnitude=4.5)


    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        'id': '20090407201255351'
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
            tags=['units:cm', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()


        print('Processing data...\n')
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model, 
            include_mt=False, include_force=True)

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
        results = results_sw

        # source index corresponding to minimum misfit
        idx = results.idxmin('source')

        best_source = grid.get(idx)
        force_dict = grid.get_dict(idx)



    #
    # Saving results
    #


    if comm.rank==0:

        plot_misfit_force(event_id+'_misfit_force.png', 
            results, title='L2 misfit')

        plot_magnitude_tradeoffs_force(event_id+'_force_tradeoffs.png',
            results, title='Magnitude tradeoffs')


        print('Plotting likelihoods...\n')

        sigma = calculate_sigma(data_sw, greens_sw,
            best_source, misfit_sw.norm, ['Z','R'],
            misfit_sw.time_shift_min, misfit_sw.time_shift_max)

        plot_likelihood_force(event_id+'_likelihood_force.png', 
            results, sigma**2, title='Maximum likelihoods')


        print('\nFinished\n')

