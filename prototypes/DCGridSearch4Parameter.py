
import numpy as np

import mtuq.io
import mtuq.misfit
import mtuq.wavelets

from mtqu import greens
from mtuq.process_data import process_bw_factory, process_sw_factory, convolve_greens
from mtuq.grid_search import MTGridRandom, grid_search_mpi


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
    'greens_functions': '/center1/ERTHQUAK/rmodrak/data/wf/FK_SYNTHETICS/scak',
    )}



if __name__=='__main__':
    """ Carries out grid search over double-couple parameters and magnitude;
       event depth and event location are fixed
    """
    # define data misfit and processing functions
    misfit = mtuq.misfit.waveform_difference_cc

    process_data = {
       'bw': process_bw_factory(**parameters_bw),
       'sw': process_sw_factory(**parameters_sw),
       }

    # read data
    data = mtuq.io.read(path, data_format)
    origin = mtuq.io.get_origin(data, data_format)
    stations = mtuq.io.get_stations(data, data_format)

    # define moment tensor grid
    grid = DCGridRandom(npts=30000, Mw=np.linspace(3.5,4.5,0.1))

    # read Green's functions
    factory = greens.fk.factory(paths.greens_functions)
    greens = factory(stations, origin)
    wavelet = mtuq.wavelets.trapezoid(half_duration=1.)
    greens.convolve(wavelet)

    # data processing
    categories = process_data.keys()
    processed_data = {}
    processed_greens = {}
    for key in categories:
        processed_data[key] = process_data[key](data)
        processed_greens[key] = greens.process(process_data[key])

    # carry out grid search in parallel
    grid_search_mpi(processed_data, processed_greens, misfit, grid, origin)


