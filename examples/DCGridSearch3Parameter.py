
import sys
import numpy as np

import mtuq.io
import mtuq.greens.fk
import mtuq.misfit

from mtuq.process_data import process_bw_factory, process_sw_factory
from mtuq.grid_search import DCGridRandom, grid_search
from mtuq.util.util import Struct
from mtuq.util.wavelets import trapezoid


parameters_bw = {
    'period_min':20.,
    'period_max':40.,
    'window_length_seconds':100.,
    }

parameters_sw = {
    'period_min':60.,
    'period_max':120.,
    'window_length_seconds':100.,
    }

paths = Struct({
    'data': '/u1/uaf/rmodrak/packages/capuaf/20090407201255351',
    'greens': '/center1/ERTHQUAK/rmodrak/data/wf/FK_SYNTHETICS/scak',
    })


if __name__=='__main__':
    """ Carries out grid search over double-couple moment tensor parameters;
       magnitude, event depth, and event location are fixed
    """
    # define data misfit
    misfit = {
        'body_waves': mtuq.misfit.waveform_difference,
        'surface_waves': mtuq.misfit.waveform_difference,
        }

    # define data processing
    process_data = {
       'body_waves': process_bw_factory(**parameters_bw),
       'surface_waves': process_sw_factory(**parameters_sw),
       }

    # define grid
    grid = DCGridRandom(points_per_axis=10, Mw=4.0)

    # read and process data
    data_format = 'sac'
    data = mtuq.io.read(data_format, paths.data, wildcard='*.[zrt]')
    origin = mtuq.io.get_origin(data_format, data)
    stations = mtuq.io.get_stations(data_format, data)

    processed_data = {}
    for key in process_data:
        processed_data[key] = process_data[key](data)

    # read and process Green's functions
    factory = mtuq.greens.fk.GreensTensorFactory(paths.greens)
    greens = factory(stations, origin)
    wavelet = trapezoid(rise_time=1., delta=stations[0].delta)
    greens.convolve(wavelet)

    processed_greens = {}
    for key in process_data:
        processed_greens[key] = greens.process(process_data[key])

    # carry out grid search
    grid_search(processed_data, processed_greens, misfit, grid)
