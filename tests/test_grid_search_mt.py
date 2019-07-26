
import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.serial import grid_search
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens
from mtuq.util import path_mtuq



if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK:
    #
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351
    #
    # Note however that CAP uses a different method for defining regular grids
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


    path_greens=  join(path_mtuq(), 'data/tests/benchmark_cap/greens/scak')
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
    # Following obspy, we use the variable name "source" for the mechanism of
    # an event and "origin" for the location of an event
    #

    sources = DoubleCoupleGridRegular(
        npts_per_axis=5,
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
    origin = data.get_preliminary_origins()[0]


    print 'Processing data...\n'
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print 'Reading Greens functions...\n'
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts nows
    #

    print 'Evaluating body wave misfit...\n'

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, sources)

    print 'Evaluating surface wave misfit...\n'

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, sources)

    best_misfit = (results_bw + results_sw).min()
    best_source = sources.get((results_bw + results_sw).argmin())


    best_misfit = (results_bw + results_sw).min()
    best_source = sources.get((results_bw + results_sw).argmin())

    if run_figures:
        plot_data_greens(event_name+'.png',
            [data_bw, data_sw], [greens_bw, greens_sw],
            [process_bw, process_sw], [misfit_bw, misfit_sw], 
            best_source, origin)

        plot_beachball(event_name+'_beachball.png', best_source)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-8):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            for _a, _b, _bool in zip(
                a, b, np.isclose(a, b, atol=atol, rtol=rtol)):

                print '%s:  %.e <= %.1e + %.1e * %.1e' %\
                    ('passed' if _bool else 'failed', abs(_a-_b), atol, rtol, abs(_b))
            print ''

            return np.all(
                np.isclose(a, b, atol=atol, rtol=rtol))

        if not isclose(
            best_source,
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

        print 'SUCCESS\n'
