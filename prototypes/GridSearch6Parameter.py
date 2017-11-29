
# NOT IMPLEMENTED
#   get_stations
#   wavelets.trapezoid

import numpy as np

import mtuq.io
import mtuq.misfit
import mtuq.greens
import mtuq.wavelets

from mtuq.process_data import process_bw_factory, process_sw_factory, convolve_greens
from mtuq.grid_search import MTGridRandom, grid_search_mpi



def GridSearch6Parameters(path, data_format, db_type, db_path):
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

    # define data categories and processing functions
    process_data = {
       'bw': process_bw_factory(**parameters_bw),
       'sw': process_sw_factory(**parameters_sw),
       }

    # define data misfit
    misfit = mtuq.misfit.waveform_difference_cc

    # read data
    data = mtuq.io.read(path, data_format)
    origin = mtuq.io.get_origin(data, data_format)
    meta = mtuq.io.get_stations(data, data_format)

    # generate Green's functions
    generator = mtuq.greens1d.factory(db_type, db_path)
    greens = generater(meta, origin)

    # convolve source wavelet
    half_duration = 1.
    wavelet = mtuq.wavelets.trapezoid(half_duration)
    greens.convolve(wavelet)

    # data processing
    categories = process_data.keys()
    processed_data = {}
    processed_greens = {}
    for key in categories:
        processed_data[key] = process_data[key](data)
        processed_greens[key] = greens.process(process_data[key])

    # define moment tensor grid
    gridsize = 1
    type = 'Tape2015'
    bounds = ({
        'v': [vmin, vmax],
        'w': [wmin, wmax],
        'h': [hmin, hmax],
        'delta': [dmin, dmax],
        'theta': [tmin, tmax],
        'kappa': [kmin, kmax],
        })
    grid = MTGridRandom(gridsize, type, bounds)

    # carry out grid search in parallel
    grid_search_mpi(processed_data, processed_greens, misfit, grid)


# debugging
if __name__=='__main__':
    GridSearch6Parameters(
        '/u1/uaf/rmodrak/packages/capuaf/20090407201255351',
        'sac',
        'fk',
        '/center1/ERTHQUAK/rmodrak/data/wf/FK_SYNTHETICS/scak',
        )


