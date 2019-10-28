#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.graphics import plot_data_greens, misfit_vs_depth
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import Trapezoid



if __name__=='__main__':
    #
    # Double-couple inversion example
    #   
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # This is the most complicated example. For much simpler one, see
    # SerialGridSearch.DoubleCouple.py
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

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.3, 4.4, 4.5,     
         4.6, 4.7, 4.8]) 

    depths = np.array(
         # depth in meters
        [25000, 30000, 35000, 40000,                    
         45000, 50000, 55000, 60000])         

    sources = DoubleCoupleGridRegular(
        npts_per_axis=25,
        magnitude=magnitudes)

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # The main I/O work starts now
    #

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.Get_size()

    if rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        stations = data.get_stations()
        origin = data.get_origins()[0]

        origins = []
        for depth in depths:
            origins += [origin.copy()]
            setattr(origins[-1], 'depth_in_m', depth)

        print('Reading Green''s functions...\n')
        greens = download_greens_tensors(stations, origins, model)

        print('Processing Green''s functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        origins = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    origins = comm.bcast(origins, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if rank==0:
        print('Evaluating body wave misfit...\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, sources)

    if rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, sources)

    if rank==0:
        results = results_bw + results_sw
        best_misfit = (results).min()

        _i, _j = np.unravel_index(np.argmin(results), results.shape)
        best_source = sources.get(_i)
        best_origin = origins[_j]


    #
    # Saving results
    #

    if comm.rank==0:
        print('Saving results...\n')

        plot_data_greens(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, best_origin, best_source)

        misfit_vs_depth(event_name+'_misfit_vs_depth_bw.png',
            data_bw, misfit_bw, origins, sources, results_bw)

        misfit_vs_depth(event_name+'_misfit_vs_depth_sw.png',
            data_sw, misfit_sw, origins, sources, results_sw)

        print('Finished\n')
