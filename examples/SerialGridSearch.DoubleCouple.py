#!/usr/bin/env python

import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search.serial import grid_search_mt
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens_mt
from mtuq.util import path_mtuq



if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 10 minutes. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
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


    #
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #

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

    grid = DoubleCoupleGridRandom(
        npts=50000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # The main I/O work starts now
    #

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

    print 'Reading Greens functions...\n'
    greens = get_greens_tensors(stations, origins, model=model)

    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw, stations, origins)
    greens_sw = greens.map(process_sw, stations, origins)


    #
    # The main computational work starts nows
    #

    print 'Carrying out grid search...\n'

    results = grid_search_mt(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, verbose=True)


    best_mt = grid.get(results.argmin())

    plot_data_greens_mt(event_name+'.png',
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], best_mt)

    plot_beachball(event_name+'_beachball.png', best_mt)

    print 'Finished\n'

