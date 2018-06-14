
import os
import sys
import numpy as np
import mtuq.dataset.sac
import mtuq.greens_tensor.fk

from os.path import basename, join
from mtuq.grid_search import DCGridRandom
from mtuq.grid_search import grid_search_serial
from mtuq.misfit.cap import misfit
from mtuq.process_data.cap import process_data
from mtuq.util.cap_util import trapezoid_rise_time, Trapezoid
from mtuq.util.plot import plot_waveforms
from mtuq.util.util import cross, root



if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   python GridSearchDC3Serial.py
    #
    # A typical runtime is about 60 minutes. For faster results, try 
    # GridSearchDC3.py, which runs the same inversion in parallel rather than
    # serial
    #


    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake
    #

    path_data=    join(root(), 'tests/data/20090407201255351')
    path_weights= join(root(), 'tests/data/20090407201255351/weights.dat')
    # Fow now this path exists only in my personal environment.  Eventually, 
    # we need to include it in the repository or make it available for download
    path_greens=  join(os.getenv('CENTER1'), 'data/wf/FK_SYNTHETICS/scak')
    event_name = '20090407201255351'


    #
    # Body- and surface-wave data are processed separately and held separately 
    # in memory
    #

    process_bw = process_data(
        filter_type='Bandpass',
        freq_min= 0.25,
        freq_max= 0.667,
        pick_type='from_fk_database',
        fk_database=path_greens,
        window_type='cap_bw',
        window_length=15.,
        padding_length=2.,
        weight_type='cap_bw',
        weight_file=path_weights,
        )

    process_sw = process_data(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_fk_database',
        fk_database=path_greens,
        window_type='cap_sw',
        window_length=150.,
        padding_length=10.,
        weight_type='cap_sw',
        weight_file=path_weights,
        )

    process_data = {
       'body_waves': process_bw,
       'surface_waves': process_sw,
       }


    #
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #

    misfit_bw = misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )

    misfit = {
        'body_waves': misfit_bw,
        'surface_waves': misfit_sw,
        }


    #
    # Here we specify the source parameter grid
    #

    grid = DCGridRandom(
        npts=50000,
        Mw=4.5)

    rise_time = trapezoid_rise_time(Mw=4.5)
    wavelet = Trapezoid(rise_time)



    #
    # The main I/O work starts now
    #

    print 'Reading data...\n'
    data = mtuq.dataset.sac.reader(path_data, wildcard='*.[zrt]')
    data.sort_by_distance()

    stations  = []
    for stream in data:
        stations += [stream.station]
    origin = data.get_origin()


    print 'Processing data...\n'
    processed_data = {}
    for key in ['body_waves', 'surface_waves']:
        processed_data[key] = data.map(process_data[key])
    data = processed_data


    print 'Reading Greens functions...\n'
    factory = mtuq.greens_tensor.fk.GreensTensorFactory(path_greens)
    greens = factory(stations, origin)


    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    processed_greens = {}
    for key in ['body_waves', 'surface_waves']:
        processed_greens[key] = greens.map(process_data[key])
    greens = processed_greens


    #
    # The main computational work starts nows
    #

    print 'Carrying out grid search...\n'
    results = grid_search_serial(data, greens, misfit, grid)


    print 'Saving results...\n'
    #grid.save(event_name+'.h5', {'misfit': results})
    best_mt = grid.get(results.argmin())


    print 'Plotting waveforms...\n'
    synthetics = {}
    for key in ['body_waves', 'surface_waves']:
        synthetics[key] = greens[key].get_synthetics(best_mt)
    plot_waveforms(event_name+'.png', data, synthetics, misfit)


