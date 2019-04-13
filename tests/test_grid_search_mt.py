
import os
import sys
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.serial import grid_search_mt
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.util.plot import plot_beachball, plot_data_greens_mt
from mtuq.util.util import path_mtuq



if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
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


    path_greens=  join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/greens/scak')
    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
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


    grid = DoubleCoupleGridRegular(
        magnitude=4.5, 
        npts_per_axis=5)

    wavelet = Trapezoid(
        magnitude=4.5)



    #
    # The integration test starts now
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
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origins)


    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw, stations, origins)
    greens_sw = greens.map(process_sw, stations, origins)


    print 'Carrying out grid search...\n'

    results = grid_search_mt(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, verbose=False)

    best_mt = grid.get(results.argmin())

    # reevalute misfit to attach time shifts
    _ = misfit_bw(data_bw, greens_bw, best_mt)
    _ = misfit_sw(data_sw, greens_sw, best_mt)

    if run_figures:
        plot_data_greens_mt(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, best_mt,
            annotate=True, stations=stations)

        plot_beachball(event_name+'_beachball.png', best_mt)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-8):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            result = np.isclose(a, b, atol=atol, rtol=rtol)

            print ''
            print 'Debugging information:'
            print ''
            for _a, _b in zip(a,b):
                print '  %.e <= %.1e + %.1e * %.1e' %\
                     (abs(_a-_b), atol, rtol, abs(_b))
            print ''
            for boolean in result:
                print '  %s' %  boolean
            print ''

            return np.all(result)

        if not isclose(
            best_mt,
            np.array([
                -1.92678437e+15,
                -1.42813064e+00,
                 1.92678437e+15,
                 2.35981928e+15,
                 6.81221149e+14,
                 1.66864422e+15,
                 ])
            ):
            raise Exception(
                "Grid search result differs from previous mtuq result")

