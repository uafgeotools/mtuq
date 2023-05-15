#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.stochastic_sampling import initialise_mt
from mtuq.stochastic_sampling.cmaes_parallel import parallel_CMA_ES
import matplotlib.pyplot as plt
from mtuq.util.signal import to_gamma, to_delta

def plot_lune(CMA):
    A = CMA.mutants_logger_list
    LIST = A.copy()
    v = LIST['v']
    w = LIST['w']
    m = LIST['misfit']
    V,W = CMA._datalogger(mean=True)['v'], CMA._datalogger(mean=True)['w']

    plt.scatter(to_gamma(v), to_delta(w), c=np.log(m))
    plt.scatter(to_gamma(V), to_delta(W), c='red', marker='x', zorder=10000000)
    plt.show()


if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   


    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'
    mode =        'greens' # 'database' or 'greens'
    # mode =        'database' # 'database' or 'greens'

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
        time_shift_min=-10.,
        time_shift_max=+10.,
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

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
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
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\n')
        if mode == 'greens':
            db = open_db('/Users/julienthurin/Downloads/model/ak135f/', format='AxiSEM')
            greens = db.get_greens_tensors(stations, origin, model)
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
        db = open_db('/Users/julienthurin/Downloads/model/ak135f/', format='AxiSEM')

    #
    # The main computational work starts now
    #
    if mode == 'database':
        parameter_list = initialise_mt(Mw_range=[4,6], depth_range=[30000, 55000])
    elif mode == 'greens':
        parameter_list = initialise_mt(Mw_range=[4,6])
        parameter_list = initialise_mt(Mw_range=[4,6])

    DATA = [data_bw, data_sw]
    PROCESS = [process_bw, process_sw]
    MISFIT = [misfit_bw, misfit_sw]

    CMA = parallel_CMA_ES(parameter_list , origin=origin, lmbda=240)
    CMA.sigma = 0.2
    iter = 10
    for i in range(iter):
        if comm.rank==0:
            print('Iteration %d\n' % i)
        CMA.draw_mutants()
        if mode == 'database':
            mis_bw = CMA.eval_fitness(data_bw, stations, misfit_bw, db, origin,  process_bw, wavelet, verbose=False)
            mis_sw = CMA.eval_fitness(data_sw, stations, misfit_sw, db, origin,  process_sw, wavelet, verbose=False)
        elif mode == 'greens':
            mis_bw = CMA.eval_fitness(data_bw, stations, misfit_bw, greens_bw)
            mis_sw = CMA.eval_fitness(data_sw, stations, misfit_sw, greens_sw)
        CMA.gather_mutants()
        CMA.fitness_sort(mis_bw+mis_sw)
        CMA.update_mean()
        CMA.update_step_size()
        CMA.update_covariance()
        # -- WORK IN PROGRESS --
        # Debug plot

        # plot_lune(CMA)
        # plt.pause(0.01)
        # -- END OF WORK IN PROGRESS --

        # if i = 0 or multiple of 10 and Last iteration:
        # if i == 0 or i % 10 == 0 or i == iter-1:
        #     CMA.plot_mean_waveforms(DATA, PROCESS, MISFIT, stations, db)
        
        # if i == 0 or Last iteration:
        if i == 0 or i == iter-1:
            CMA.plot_mean_waveforms(DATA, PROCESS, MISFIT, stations, db)