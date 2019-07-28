#!/usr/bin/env python

import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search.mpi import grid_search
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens
from mtuq.util import path_mtuq



if __name__=='__main__':
    #
    # THIS EXAMPLE ONLY WORKS ON CHINOOK.ALASKA.EDU
    #

    #
    # CAP-style double-couple inversion example
    # 

    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors, using Green's functions and phase picks from a local
    # FK database

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

    sources = DoubleCoupleGridRandom(
        npts=50000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)


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
        origin = data.get_preliminary_origins()[0]

        print 'Processing data...\n'
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print 'Reading Greens functions...\n'
        db = open_db(path_greens, format='FK', model=model)
        greens = db.get_greens_tensors(stations, origins)

        print 'Processing Greens functions...\n'
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        origin = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    origin = comm.bcast(origin, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print 'Evaluating body wave misfit...\n'

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, sources)

    if comm.rank==0:
        print 'Evaluating surface wave misfit...\n'

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, sources)

    results_bw = comm.gather(results_bw, root=0)
    results_sw = comm.gather(results_sw, root=0)

    if comm.rank==0:
        results_bw = np.concatenate(results_bw, axis=1)
        results_sw = np.concatenate(results_sw, axis=1)

        best_misfit = (results_bw + results_sw).min()
        best_source = sources.get((results_bw + results_sw).argmin())


    #
    # Saving results
    #

    if comm.rank==0:
        print 'Savings results...\n'

        plot_data_greens(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source)

        plot_beachball(event_name+'_beachball.png', best_source)

        #grid.save(event_name+'.h5', {'misfit': results})

        print 'Finished\n'

