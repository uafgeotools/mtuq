
import os
import sys
import numpy as np

from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.mpi import grid_search_serial
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.util.plot import plot_beachball, plot_data_greens_mt
from mtuq.util.util import path_mtuq



if __name__=='__main__':
    #
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple3.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351


    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135'


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

    process_data = {
       'body_waves': process_bw,
       'surface_waves': process_sw,
       }


    misfit_bw = Misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )

    misfit = {
        'body_waves': misfit_bw,
        'surface_waves': misfit_sw,
        }


    grid = DoubleCoupleGridRegular(
        moment_magnitude=4.5, 
        npts_per_axis=10)

    wavelet = Trapezoid(
        moment_magnitude=4.5)



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

    print 'Downloading Greens functions...\n'
    greens = get_greens_tensors(stations, origins[0], model=model)


    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw, stations, origins)
    greens_sw = greens.map(process_sw, stations, origins)


    processed_data = {
         'body_waves': data_bw,
         'surface_waves': data_sw,
         }

    processed_greens = {
         'body_waves': greens_bw,
         'surface_waves': greens_sw,
         }

    misfit = {
         'body_waves': misfit_bw,
         'surface_waves': misfit_sw,
         }

    #
    # The main computational work starts nows
    #

    print 'Carrying out grid search...\n'

    results = grid_search_serial(
         processed_data, processed_greens, misfit, grid)

    best_mt = grid.get(results.argmin())

    plot_data_greens_mt(event_name+'.png',
        processed_data, processed_greens, best_mt, misfit)

    plot_beachball(event_name+'_beachball.png',
        best_mt)


