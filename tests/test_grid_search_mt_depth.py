
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
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we included mangitude and depth and use a coarser grid
    #

    # by default, the script runs with figure generation and error checking
    # turned on
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)


    path_greens=  fullpath('data/tests/benchmark_cap/greens/scak')
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'


    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='FK_metadata',
        FK_database=path_greens,
        window_type='body_wave',
        window_length=15.,
        padding_length=2.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='FK_metadata',
        FK_database=path_greens,
        window_type='surface_wave',
        window_length=150.,
        padding_length=10.,
        capuaf_file=path_weights,
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

    depths = np.array(
         # depth in meters
        [34000])

    sources = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitude=[4.4, 4.5, 4.6])

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # The main I/O work starts now
    #

    print('Reading data...\n')
    data = read(path_data, format='sac', 
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]

    origins = []
    for depth in depths:
        origins += [origin.copy()]
        setattr(origins[-1], 'depth_in_m', depth)

    print('Processing data...\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Green''s functions...\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origins)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, sources, verbose=False)

    print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, sources, verbose=False)



    best_misfit = (results_bw + results_sw).min()
    best_source = sources.get((results_bw + results_sw).argmin())

    if run_figures:
        filename = event_name+'_misfit_vs_depth.png'
        #misfit_vs_depth(filename, best_misfit)

    if run_checks:
        pass

    print('SUCCESS\n')

