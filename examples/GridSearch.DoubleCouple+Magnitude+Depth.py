#!/usr/bin/env python

import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.mpi import grid_search_mt_depth
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import beachball_vs_depth, misfit_vs_depth
from mtuq.graphics.waveform import plot_data_greens_mt
from mtuq.util import path_mtuq



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

    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
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
        pick_type='from_taup_model',
        taup_model=model,
        window_type='cap_bw',
        window_length=15.,
        padding_length=2.,
        weight_type='cap_bw',
        cap_weight_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_taup_model',
        taup_model=model,
        window_type='cap_sw',
        window_length=150.,
        padding_length=10.,
        weight_type='cap_sw',
        cap_weight_file=path_weights,
        )


    misfit_bw = Misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # Next we specify the source parameter grid
    #

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.3, 4.4, 4.5,     
         4.6, 4.7, 4.8]) 

    depths = np.array(
         # depth in meters
        [25000, 30000, 35000, 40000,                    
         45000, 50000, 55000, 60000])         

    grid = DoubleCoupleGridRegular(
        npts_per_axis=25,
        magnitude=magnitudes)

    wavelet = Trapezoid(
        magnitude=np.mean(magnitudes))


    #
    # The main I/O work starts now
    #

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.Get_size()

    if rank==0:
        print 'Reading data...\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origins = data.get_origins()


        print 'Processing data...\n'
        data_bw = data.map(process_bw, stations, origins)
        data_sw = data.map(process_sw, stations, origins)

    else:
        data_bw = None
        data_sw = None

    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)

    greens_bw = {}
    greens_sw = {}

    if rank==0:
        print 'Reading Greens functions...\n'

        for _i, depth in enumerate(depths):
            print '  Depth %d of %d' % (_i+1, len(depths))

            origins = deepcopy(origins)
            [setattr(origin, 'depth_in_m', depth) for origin in origins]

            greens = get_greens_tensors(stations, origins, model=model)

            greens.convolve(wavelet)
            greens_bw[depth] = greens.map(process_bw, stations, origins)
            greens_sw[depth] = greens.map(process_sw, stations, origins)

    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if rank==0:
        print 'Carrying out grid search...\n'

    results = grid_search_mt_depth(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, depths)

    # gathering results
    results_unsorted = comm.gather(results, root=0)
    if rank==0:
        results = {}
        for depth in depths:
            results[depth] = np.concatenate(
                [results_unsorted[iproc][depth] for iproc in range(nproc)])

    if comm.rank==0:
        print 'Saving results...\n'

        best_misfit = {}
        best_mt = {}
        for depth in depths:
            best_misfit[depth] = results[depth].min()
            best_mt[depth] = grid.get(results[depth].argmin())

        filename = event_name+'_beachball_vs_depth.png'
        beachball_vs_depth(filename, best_mt)

        filename = event_name+'_misfit_vs_depth.png'
        misfit_vs_depth(filename, best_misfit)

        print 'Finished\n'
