
import sys
import numpy as np

import mtuq.io
import mtuq.greens.fk
import mtuq.misfit

from mtuq.process_data import process_bw_factory, process_sw_factory
from mtuq.grid_search import MTGridRandom, grid_search_mpi
from mtuq.util.util import Struct
#from mtuq.util.wavelets import trapezoid


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
    """ Carries out grid search over double-couple parameters;
       magnitude, event depth, and event location are fixed
    """
    # define data misfit
    misfit = mtuq.misfit.waveform_difference_cc

    # define data processing
    process_data = {
       'bw': process_bw_factory(**parameters_bw),
       'sw': process_sw_factory(**parameters_sw),
       }

    # read data
    data_format = 'sac'
    data = mtuq.io.read(data_format, paths.data)
    origin = mtuq.io.get_origin(data_format, data)
    stations = mtuq.io.get_stations(data_format, data)

    # read Green's functions
    factory = mtuq.greens.fk.GreensTensorFactory(paths.greens)
    greens = factory(stations, origin)
    #wavelet = trapezoid(half_duration=1.)
    #greens.convolve(wavelet)

    # data processing
    categories = process_data.keys()
    processed_data = {}
    processed_greens = {}
    for key in categories:
        processed_data[key] = process_data[key](data)
        processed_greens[key] = greens.process(process_data[key])

    # define grid
    grid = DCGridRandom(npts=30000, Mw=4.0)

    # carry out grid search in parallel
    grid_search_mpi(processed_data, processed_greens, misfit, grid, origin)


