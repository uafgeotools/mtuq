
import numpy as np

import mtuq.io
import mtuq.misfit_functions
import mtuq.synthetics1d

from mtuq.process_data import bw_factory, sw_factory
from mtuq.grid_search import Grid, UniformGrid, grid_search


if __name__ == '__main__':
    # event information
    path = ''
    name = ''
    data_format = 'sac'

    # define data categories and data processing functions
    process_data {
       'bw': process_bw_factory(Tmin=20., Tmax=40.),
       'sw': process_sw_factory(Tmin=60., Tmax=120.),
       }

    # define data misfit
    misfit = mtuq.misfit_functions('waveform_difference',
        normalize='L2',
        station_correction='cc')

    # define Green's functions generater
    gf_generater = mtuq.synthetics1d.fk_factory(path='')

    # read data
    data = mtuq.io.read(path, format=data_format)
    origin = mtuq.io.get_origin(data, format=data_format)
    stations = mtuq.io.get_stations(data, format=data_format)

    # preload Greens functions
    green_functions = generator(data, origin, stations)
    convolved = convolve_source_wavelet(greens_functions, wavelet)

    # process traces
    categories = process_data.keys()
    processed_data = {}
    processed_greens_functions = {}
    for key in categories:
        processed_data[key] = process_data[key](data)
        processed_greens_functions[key] = process_data[key](convolved_greens_functions)

    # define moment tensor grid
    mt_type = 'Tape2015'
    mt_bounds = ({
        'v': [vmin, vmax, nv],
        'w': [wmin, wmax, nw],
        'h': [hmin, hmax, nw],
        'delta': [min, max, n],
        'theta': [min, max, n],
        'kappa': [min, max, n],
        })
    grid = MTGridUniform(mt_type, mt_bounds)

    # carry out grid search in parallel
    grid_search_mt(processed_data, processed_greens_functions, misfit, grid)

