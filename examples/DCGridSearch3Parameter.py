
import os
import sys
import numpy as np
import mtuq.dataset.sac
import mtuq.greens_tensor.fk

from os.path import basename, join
from mtuq.grid_search import DCGridRandom, grid_search_serial
from mtuq.misfit.legacy import cap_bw, cap_sw
from mtuq.process_data.cap import process_data
from mtuq.util.plot import cap_plot
from mtuq.util.util import AttribDict, root
from mtuq.util.wavelets import trapezoid


if __name__=='__main__':
    """
     Double-couple inversion example

    Carries out a grid search over double-couple moment tensor parameters,
    with magnitude, depth, and location fixed

    A typical runtime is about 60 minutes. For faster results, try
    mtuq/examples/DCGridSearchMPI.py
    """

    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake. For now, these paths exist only in my personal 
    # environment.  Eventually we need to include sample data in the 
    # repository or make it available for download
    #
    paths = AttribDict({
        'data':    join(root(), 'tests/data/20090407201255351'),
        'weights': join(root(), 'tests/data/20090407201255351/weight_test.dat'),
        'greens':  os.getenv('CENTER1')+'/'+'data/wf/FK_SYNTHETICS/scak',
        })


    #
    # Here we specify all the data processing and misfit settings used in the
    # for the inversion.  In this example, body- and surface-waves are 
    # processed separately, and misfit is defined as a sum of indepdendent 
    # body- and surface-wave contributions. (For a more flexible way of
    # of specifying parameters based on command line argument passing, see
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
        window_length=150.,
        #window_type='cap_sw',
        weight_type='cap_sw',
        weight_file=paths.weights,
        )

    process_data = {
       'body_waves': process_bw,
       'surface_waves': process_sw,
       }


    misfit_bw = cap_bw(
        max_shift=2.,
        )

    misfit_sw = cap_sw(
        max_shift=10.,
        )

    misfit = {
        'body_waves': misfit_bw,
        'surface_waves': misfit_sw,
        }

    grid = DCGridRandom(
        npts=50000,
        Mw=4.5,
        )


    #
    # The computational work of the grid search begins now
    #

    print 'Reading data...\n'
    data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
    origin = data.get_origin()
    stations = data.get_stations()
    event_name = data.id


    print 'Processing data...\n'
    processed_data = {}
    for key in process_data:
        processed_data[key] = data.map(process_data[key], stations)
    data = processed_data


    print 'Reading Greens functions...\n'
    generator = mtuq.greens_tensor.fk.Generator(paths.greens)
    greens = generator(stations, origin)


    print 'Processing Greens functions...\n'
    greens.convolve(trapezoid(rise_time=1.))
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

