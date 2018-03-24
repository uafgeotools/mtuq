
import os
import sys
import numpy as np
import mtuq.dataset.sac
import mtuq.greens_tensor.fk

from os.path import basename, join
from mtuq.grid_search import DCGridRandom, grid_search_serial
from mtuq.misfit.legacy import misfit
from mtuq.process_data.cap import process_data
from mtuq.util.geodetics import cap_rise_time
from mtuq.util.plot import cap_plot
from mtuq.util.util import AttribDict, root
from mtuq.util.wavelets import Trapezoid


if __name__=='__main__':
    """
    Double-couple inversion example

    Carries out a grid search over 50,000 randomly chosen double-couple 
    moment tensors; magnitude, depth, and location kept fixed

    A typical runtime is about 60 minutes. For faster results, try
    mtuq/examples/DCGridSearch3ParameterMPI.py
    """

    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4.5 Alaska earthquake. For now, these paths exist only in my personal 
    # environment--eventually we need to include sample data in the 
    # repository or make it available for download
    #
    paths = AttribDict({
        'data':    join(root(), 'tests/data/20090407201255351'),
        'weights': join(root(), 'tests/data/20090407201255351/weight_test.dat'),
        'greens':  join(os.getenv('CENTER1'), 'data/wf/FK_SYNTHETICS/scak'),
        })

    event_name = '20090407201255351'


    #
    # Here we specify all the data processing and misfit settings used in the
    # inversion.  For this example, body- and surface-waves are processed
    # separately, and misfit is a sum of indepdendent body- and surface-wave
    # contributions. (For a more flexible way of specifying parameters based on
    # command-line argument passing rather than scripting, see
    # mtuq/scripts/cap_inversion.py)
    #

    process_bw = process_data(
        filter_type='Bandpass',
        freq_min= 0.25,
        freq_max= 0.667,
        window_type='cap_bw',
        window_length=15.,
        weight_type='cap_bw',
        weight_file=paths.weights,
        )

    process_sw = process_data(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        #window_type='cap_sw',
        window_length=150.,
        weight_type='cap_sw',
        weight_file=paths.weights,
        )

    process_data = {
       'body_waves': process_bw,
       'surface_waves': process_sw,
       }


    misfit_bw = misfit(
        time_shift_max=2.,
        )

    misfit_sw = misfit(
        time_shift_max=10.,
        )

    misfit = {
        'body_waves': misfit_bw,
        'surface_waves': misfit_sw,
        }


    #
    # Here we specify the moment tensor grid and source wavelet
    #

    grid = DCGridRandom(
        npts=50000,
        Mw=4.5,
        )

    rise_time = cap_rise_time(Mw=4.5)
    wavelet = Trapezoid(rise_time)


    #
    # The computational work of the grid search begins now
    #

    print 'Reading data...\n'
    data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
    origin = data.get_origin()
    stations = data.get_stations()


    print 'Processing data...\n'
    processed_data = {}
    for key in process_data:
        processed_data[key] = data.map(process_data[key], stations)
    data = processed_data


    print 'Reading Greens functions...\n'
    generator = mtuq.greens_tensor.fk.Generator(paths.greens)
    greens = generator(stations, origin)


    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    processed_greens = {}
    for key in process_data:
        processed_greens[key] = greens.map(process_data[key], stations)
    greens = processed_greens


    print 'Carrying out grid search...\n'
    results = grid_search_serial(data, greens, misfit, grid)


    print 'Saving results...\n'
    grid.save(event_name+'.h5', {'misfit': results})


    print 'Plotting waveforms...\n'
    mt = grid.get(results.argmin())
    cap_plot(event_name+'.png', data, greens, mt, paths.weights)


