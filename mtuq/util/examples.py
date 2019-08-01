
import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search.serial import grid_search
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens
from mtuq.util import path_mtuq



if True:
    #
    # Creates example data structures
    #
    # Rather than being executed as a script, this code is designed to be
    # imported.  After importing this module, users can access the example data
    # and functions listed in __all__
    #
    # Note that some I/O and data processing are involved in creating the
    # example data, so importing this module may take a few seconds longer than
    # most other modules
    #
    
    __all__ = [
        'process_bw'
        'process_bw'
        'misfit_bw',
        'misfit_bw',
        'data_bw',
        'data_sw',
        'greens_bw',
        'greens_sw',
        'stations',
        'origin',
        ]

    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135'


    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_taup_model',
        taup_model=model,
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
        pick_type='from_taup_model',
        taup_model=model,
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


    #
    # Following obspy, we use the variable name "source" for the mechanism of
    # an event and "origin" for the location of an event
    #

    sources = DoubleCoupleGridRegular(
        npts_per_axis=10,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # The main I/O work starts now
    #

    
    data = read(path_data, format='sac',
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_preliminary_origins()[0]


    
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    
    greens = get_greens_tensors(stations, origin, model=model)

    
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)

