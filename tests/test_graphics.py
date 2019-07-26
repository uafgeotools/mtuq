
import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search.mpi import grid_search
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens
from mtuq.util import path_mtuq



if __name__=='__main__':
    #
    # Tests data, synthetics and beachball plotting utilities
    #
    # Note that in the figures created by this script, the data and synthetics 
    # are not expected to fit epsecially well; currently, the only requirement 
    # is that the script runs without errors
    #

    import matplotlib
    matplotlib.use('Agg', warn=False, force=True)
    import matplotlib

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


    mt = np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]) # explosion
    mt *= 1.e16

    wavelet = Trapezoid(
        magnitude=4.5)


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
    # Generate figures
    #

    print 'Figure 1 of 3\n'

    plot_data_greens(event_name+'.png',
        [data_bw, data_sw], [greens_bw, greens_sw],
        [process_bw, process_sw], [misfit_bw, misfit_sw], 
        mt, origin, header=False)

    print 'Figure 2 of 3\n'

    plot_data_greens(event_name+'.png',
        [data_bw, data_sw], [greens_bw, greens_sw],
        [process_bw, process_sw], [misfit_bw, misfit_sw], 
        mt, origin, header=False)

    print 'Figure 3 of 3\n'

    plot_beachball('test_graphics3.png', mt)

    print 'Finished\n'
