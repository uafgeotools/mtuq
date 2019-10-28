#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.graphics import plot_data_greens, plot_beachball
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import Trapezoid



if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple.py
    #
    # For a simpler example, see SerialGridSearch.DoubleCouple.py, 
    # which runs the same inversion in serial
    #


    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake
    #

    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135'


    #
    # Body- and surface-wave data are processed separately and held separately 
    # in memory
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        padding_length=2.,
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
        padding_length=10.,
        capuaf_file=path_weights,
        )


    #
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #

    misfit_bw = Misfit(
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        data_processing=process_bw,
        )

    misfit_sw = Misfit(
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        data_processing=process_sw,
        )


    #
    # Following obspy, we use the variable name "source" for the mechanism of
    # an event and "origin" for the location of an event
    #

    sources = DoubleCoupleGridRandom(
        npts=50000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origin = data.get_origins()[0]

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Green''s functions...\n')
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        origin = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    origin = comm.bcast(origin, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print('Evaluating body wave misfit...\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, sources)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, sources)

    if comm.rank==0:
        best_misfit = (results_bw + results_sw).min()
        best_source = sources.get((results_bw + results_sw).argmin())


    #
    # Saving results
    #

    if comm.rank==0:
        print('Savings results...\n')

        plot_data_greens(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source)

        plot_beachball(event_name+'_beachball.png', best_source)

        #grid.save(event_name+'.h5', {'misfit': results})

        print('Finished\n')

