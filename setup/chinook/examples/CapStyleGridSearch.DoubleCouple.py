#!/usr/bin/env python

import os
import sys
import numpy as np

from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search.mpi import grid_search_mpi
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.util.plot import plot_beachball, plot_data_greens_mt
from mtuq.util.util import path_mtuq



if __name__=='__main__':
    #
    # CAP-style Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python CapStyleGridSearch.DoubleCouple.py
    #


    path_greens= '/import/c1/ERTHQUAK/rmodrak/wf/FK_synthetics/scak'
    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'


    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_fk_metadata',
        fk_database=path_greens,
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
        pick_type='from_fk_metadata',
        fk_database=path_greens,
        window_type='cap_sw',
        window_length=150.,
        padding_length=10.,
        weight_type='cap_sw',
        cap_weight_file=path_weights,
        )


    misfit_bw = Misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # Next we specify the source parameter grid
    #

    grid = DoubleCoupleGridRandom(
        npts=50000,
        moment_magnitude=4.5)

    wavelet = Trapezoid(
        moment_magnitude=4.5)


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print 'Reading data...\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origins = data.get_origins()

        print 'Processing data...\n'
        data_bw = data.map(process_bw, stations, origins)
        data_sw = data.map(process_sw, stations, origins)

        print 'Downloading Greens functions...\n'
        db = open_db(path_greens, format='FK', model=model)
        greens = db.get_greens_tensors(stations, origins)

        print 'Processing Greens functions...\n'
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw, stations, origins)
        greens_sw = greens.map(process_sw, stations, origins)

    else:
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print 'Carrying out grid search...\n'

    results = grid_search_mpi(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid)

    results = comm.gather(results, root=0)


    if comm.rank==0:
        print 'Saving results...\n'

        results = np.concatenate(results)

        best_mt = grid.get(results.argmin())

        plot_data_greens_mt(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw,
            misfit_bw, misfit_sw, best_mt)

        plot_beachball(event_name+'_beachball.png', 
            best_mt)


