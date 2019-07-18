#!/usr/bin/env python

import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import FullMomentTensorGridRandom
from mtuq.grid_search.mpi import grid_search_mt
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens_mt
from mtuq.util import path_mtuq



if __name__=='__main__':
    #
    # Full moment tensor inversion example
    #   
    # Carries out grid search over all moment tensor parameters except
    # magnitude 
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
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


    #
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #

    misfit_bw = Misfit(
        norm='L1',
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L1',
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # Next we specify the search grid. Following obspy, we use "source" for the
    # mechanism of an event and "origin" for the location of an event
    #
    #

    sources = FullMomentTensorGridRandom(
        npts=1000000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print 'Reading data...\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origin = data.get_preliminary_origins()[0]

        print 'Processing data...\n'
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print 'Reading Greens functions...\n'
        greens = get_greens_tensors(stations, origin, model=model)

        print 'Processing Greens functions...\n'
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print 'Evaluating body wave misfit...\n'

    results_bw = grid_search_mt(
        data_bw, greens_bw, misfit_bw, grid)

    if comm.rank==0:
        print 'Evaluating surface wave misfit...\n'

    results_sw = grid_search_mt(
        data_sw, greens_sw, misfit_sw, grid)

    results_bw = comm.gather(results_bw, root=0)
    results_sw = comm.gather(results_sw, root=0)

    if comm.rank==0:
        results_bw = np.concatenate(results_bw)
        results_sw = np.concatenate(results_sw)

        best_misfit = (results_bw + results_sw).min()
        best_source = sources.get((results_bw + results_sw).argmin())


    #
    # Saving results
    #

    if comm.rank==0:
        print 'Savings results...\n'

        plot_data_greens_mt(event_name+'.png',
            [data_bw, data_sw], [greens_bw, greens_sw],
            [process_bw, process_sw], [misfit_bw, misfit_sw], 
            best_source)

        plot_beachball(event_name+'_beachball.png', best_source)

        #grid.save(event_name+'.h5', {'misfit': results})

        print 'Finished\n'

