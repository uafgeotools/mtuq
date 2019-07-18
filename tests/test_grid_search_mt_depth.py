
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
from mtuq.graphics.beachball import beachball_vs_depth, misfit_vs_depth
from mtuq.graphics.waveform import plot_data_greens_mt
from mtuq.util import path_mtuq



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

    greens_bw = {}
    greens_sw = {}

    print 'Reading Greens functions...\n'

    for _i, depth in enumerate(depths):
        origins = deepcopy(origins)
        [setattr(origin, 'depth_in_m', depth) for origin in origins]

        db = open_db(path_greens, format='FK', model=model)
        greens = db.get_greens_tensors(stations, origin)

        greens.convolve(wavelet)
        greens_bw[depth] = greens.map(process_bw)
        greens_sw[depth] = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print 'Carrying out grid search...\n'

    results = grid_search(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], sources, origins, verbose=False)


    best_misfit = {}
    best_source = {}
    for depth in depths:
        best_misfit[depth] = results[depth].min()
        best_source[depth] = sources.get(results[depth].argmin())

    if run_figures:
        filename = event_name+'_beachball_vs_depth.png'
        beachball_vs_depth(filename, best_source)

        filename = event_name+'_misfit_vs_depth.png'
        misfit_vs_depth(filename, best_misfit)

    if run_checks:
        pass

    print 'SUCCESS\n'

