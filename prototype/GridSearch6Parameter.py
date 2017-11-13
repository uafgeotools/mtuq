
import numpy as np

import mtuq.io
import mtuq.misfit_functions
import mtuq.synthetics1d

from mtuq.process_data import bw_factory, sw_factory
from mtuq.grid_search import Grid, UniformGrid, grid_search


if __name__ == '__main__':
    # event information
    path = 'dummy_path'
    name = 'dummy_name'
    data_format = 'sac'

    # read data
    data = mtuq.io.read(path, data_format)
    origin = mtuq.io.get_origin(data, data_format)
    stations = mtuq.io.get_stations(data, data_format)

    # define data categories and processing functions
    process_data = {
       'bw': process_bw_factory(Tmin=20., Tmax=40.),
       'sw': process_sw_factory(Tmin=60., Tmax=120.),
       }

    # define data misfit
    misfit = mtuq.misfit.waveform_difference_cc

    # define Green's functions generater
    model = 'dummy_model'
    gf_generater = mtuq.synthetics1d.fk_factory(model)

    # read precomputed Greens functions
    green_functions = gf_generator(data, origin, stations)
    convolved_gf = convolve_source_wavelet(greens_functions, wavelet)

    # resample Greens functions

    # process traces
    categories = process_data.keys()
    processed_data = {}
    processed_greens_functions = {}
    for key in categories:
        processed_data[key] = process_data[key](data)
        processed_gf[key] = process_data[key](convolved_gf)

    # define moment tensor grid
    mt_type = 'Tape2015'
    mt_bounds = ({
        'v': [vmin, vmax, nv],
        'w': [wmin, wmax, nw],
        'h': [hmin, hmax, nw],
        'delta': [dmin, dmax, n],
        'theta': [tmin, tmax, n],
        'kappa': [kmin, kmax, n],
        })
    grid = MTGridUniform(mt_type, mt_bounds)

    # carry out grid search in parallel
    grid_search_mt_mpi(processed_data, processed_gf, misfit, grid)


