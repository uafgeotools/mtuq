#!/usr/bin/env python

import sys
sys.path.insert(1, '/scratch/l/liuqy/liangd/workspace/DPackages/seishmc/')


import os
import numpy as np


from seishmc.DHMC.dc import DHMC_DC
from seishmc.visualization.viz_samples_dc import pairplot_samples_DC

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Carries out Hamiltonian Monte Carlo (HMC) sampling over double couple moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python HMC.DoubleCouple.py
    #
    #

    #
    # We will investigate the source process of an Mw~4.8 earthquake using data
    # from a regional seismic array
    #

    path_data   = fullpath('data/examples/SPECFEM3D_SGT/data/*.[zrt]')
    path_greens = fullpath('data/examples/SPECFEM3D_SGT/greens/socal3D')
    path_weights= fullpath('data/examples/SPECFEM3D_SGT/weights.dat')
    event_id    = 'evt11071294'
    model       = 'socal3D'
    taup_model  = 'ak135'

    # output folder
    saving_dir = './'

    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.05,
        freq_max= 0.125,
        pick_type='taup',
        taup_model=taup_model,
        window_type='body_wave',
        window_length=30.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.033333,
        freq_max=0.125,
        pick_type='taup',
        taup_model=taup_model,
        window_type='surface_wave',
        window_length=100.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-3.,
        time_shift_max=+3.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-3.,
        time_shift_max=+3.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    wavelet = Trapezoid(
        magnitude=4.8)

    #
    # Origin time and location will be fixed.
    #

    origin = Origin({
        'time': '2019-07-12T13:11:37.0000Z',
        'latitude': 35.638333,
        'longitude': -117.585333,
        'depth_in_m': 9950.0,
        'id': 'evt11071294'
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:cm', 'type:velocity'])


        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        db = open_db(path_greens, format='SPECFEM3D_SGT', model='socal3D')
        greens = db.get_greens_tensors(stations, origin)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    #
    # The main computational work starts now
    #

    rank = comm.Get_rank()
    print('Initialize HMC.\n')
    solver_hmc = DHMC_DC(misfit_bw, data_bw, greens_bw,
                            misfit_sw, data_sw, greens_sw,
                            saving_dir, b_save_cache=True,
                            n_step_cache=100, verbose=True)

    # set the range of number of step
    solver_hmc.set_n_step(min=20, max=50)

    # set the range of step interval
    solver_hmc.set_epsilon(min=0.05, max=1.0)

    # set the number of accepted samples
    n_sample = 50
    # n_sample = 500

    # set initial solution in degree and Mw
    # [strike, dip, rake, mag]
    q0 = np.array([np.random.uniform(0, 360),
                   np.random.uniform(0, 90),
                   np.random.uniform(0, 180),
                   np.random.uniform(4.5, 5.0)])
    solver_hmc.set_q(q0)

    print('Sampling ...\n')
    task_id = '%s_DC_HMC_%d' % (event_id, rank)
    solver_hmc.sampling(n_sample=n_sample, task_id=task_id)




    print('Generating figures...\n')
    data_file = os.path.join(saving_dir, "%s_samples_N%d.pkl"%(task_id, n_sample))
    fig_path = os.path.join(saving_dir, task_id)

    pairplot_samples_DC(file_path=data_file, fig_saving_path=fig_path, init_sol=q0)


    # Get the solution
    best_mt, lune_dict = solver_hmc.get_solution()

    fig_path = os.path.join(saving_dir, '%s_waveforms.png' % task_id)
    plot_data_greens2(fig_path,
                      data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
                      misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

    fig_path = os.path.join(saving_dir, '%s_beachball.png' % task_id)
    plot_beachball(fig_path, best_mt, stations, origin)

    MPI.Finalize()
    print('\nFinished\n')