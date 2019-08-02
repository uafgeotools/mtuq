#!/usr/bin/env python

import os
import numpy as np

from copy import deepcopy
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.mpi import grid_search
from mtuq.graphics import plot_data_greens, misfit_vs_depth
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import Trapezoid



if __name__=='__main__':
    #
    # THIS EXAMPLE ONLY WORKS ON CHINOOK.ALASKA.EDU
    #

    #
    # CAP-style double-couple inversion example
    # 

    #
    # Carries out grid search over source orientation, magnitude, and depth
    # using Green's functions and phase picks from a local FK database
    #

    #
    # USAGE
    #   mpirun -n <NPROC> python CapStyleGridSearch.DoubleCouple+Magntidue+Depth.py
    #


    path_greens= '/import/c1/ERTHQUAK/ERTHQUAK/FK_synthetics/scak'
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'


    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_fk_metadata',
        fk_database=path_greens,
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
        pick_type='from_fk_metadata',
        fk_database=path_greens,
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
        print 'Reading data...\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origin = data.get_origins()[0]

        origins = []
        for depth in depths:
            origins += [deepcopy(origin)]
            setattr(origins[-1], 'depth_in_m', depth)

        greens = get_greens_tensors(stations, origins, model=model)

        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

        print 'Processing data...\n'
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

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
        print 'Evaluating body wave misfit...\n'

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, sources)

    if rank==0:
        print 'Evaluating surface wave misfit...\n'

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, sources)

    # gathering results
    results_bw = comm.gather(results_bw, root=0)
    results_sw = comm.gather(results_sw, root=0)

    if rank==0:
        results_bw = np.concatenate(results_bw, axis=1)
        results_sw = np.concatenate(results_sw, axis=1)

    #
    # Saving results
    #

    if comm.rank==0:
        print 'Saving results...\n'

        filename = event_name+'_misfit_vs_depth.png'
        misfit_vs_depth(filename, results_bw+results_sw, origins, sources)

        filename = event_name+'_bw_vs_depth.png'
        misfit_vs_depth(filename, results_bw, origins, sources)

        filename = event_name+'_sw_vs_depth.png'
        misfit_vs_depth(filename, results_sw, origins, sources)

        print 'Finished\n'
