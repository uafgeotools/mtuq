#!/usr/bin/env python

import os
import sys
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.mpi import grid_search_mt_depth
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.util.plot import plot_beachball, plot_data_greens_mt
from mtuq.util.util import path_mtuq



if __name__=='__main__':
    #
    # Double-couple inversion example
    #   
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
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
        [4.3, 4.4, 4.5, 4.6, 4.7, 4.8])

    depths = np.array(
        [24])#, 26, 28, 30, 32, 34, 36, 38, 40, 42])

    grid = DoubleCoupleGridRegular(
        npts_per_axis=20,
        magnitude=magnitudes)

    wavelet = Trapezoid(
        magnitude=np.mean(magnitudes))


    #
    # The main I/O work starts now
    #

    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    if comm.rank==0:
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

    if comm.rank==0:
        print 'Downloading Greens functions...\n'

        for _i, depth in enumerate(depths):
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

    if comm.rank==0:
        print 'Carrying out grid search...\n'

    results = grid_search_mt_depth(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, depths)

    results = [comm.gather(results, root=0)]


    if comm.rank==0:
        print 'Saving results...\n'

        for depth in depths:
            results[depth] = np.concatenate(results[depth])

        plot_depth_test(event_name+'_depth_test.png', 
            grid, results)


