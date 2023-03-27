#!/usr/bin/env python

import os, shutil
import numpy as np

import mtuq.grid_search
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_misfit_depth
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.graphics.uq import omega
from mtuq.util.math import to_mij


result_All = mtuq.grid_search.MTUQDataFrame()
list_dir = ['pdf','cdf','rho_vs_v', 'prob_vs_omega', 'misfit_vs_omega', 'waveforms']
for items in list_dir:
    if os.path.exists(items):
        shutil.rmtree(items)
    os.makedirs(items)

if __name__=='__main__':
    #
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # For simpler examples, see SerialGridSearch.DoubleCouple.py or
    # GridSearch.FullMomentTensor.py
    #   


    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #

    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'


    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # We will search over a range of locations about the catalog origin
    #


    catalog_origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })
        
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.5]) 

    grid = DoubleCoupleGridRandom(
        npts=64000,
        magnitudes=magnitudes)

    wavelet = Trapezoid(
        magnitude=4.5)

    depths = np.array(
         # depth in meters
        [25000., 30000., 35000., 40000.,                    
         45000., 50000., 55000., 60000.])
         
    origins1 = []
    for depth in depths:
        origins1 += [catalog_origin.copy()]
        setattr(origins1[-1], 'depth_in_m', depth)
    min_misfit = []
    min_misfit_index = []
    unc = []
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    for i in range(len(depths)):
        depths_uq = [depths[i]]

        origins = []
        for depth in depths_uq:
            origins += [catalog_origin.copy()]
            setattr(origins[-1], 'depth_in_m', depth)


        #
        # The main I/O work starts now
        #

        if comm.rank == 0:
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
            greens = download_greens_tensors(stations, origins, model)

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

        if comm.rank == 0:
            print('Evaluating body wave misfit...\n')

        results_bw = grid_search(
            data_bw, greens_bw, misfit_bw, origins, grid)

        if comm.rank == 0:
            print('Evaluating surface wave misfit...\n')

        results_sw = grid_search(
            data_sw, greens_sw, misfit_sw, origins, grid)



        if comm.rank == 0:

            results = results_bw + results_sw

            origin_idx = results.origin_idxmin()
            best_origin = origins[origin_idx]


            source_idx = results.source_idxmin()
            best_mt = grid.get(source_idx)
            lune_dict = grid.get_dict(source_idx)
            mt_dict = best_mt.as_dict()


            #
            # Generate figures and save results
            #

            print('Generating figures...\n')

            plot_data_greens2('waveforms/' + event_id + 'DC+Z_waveforms' +str(depths[i])+ '.png',
                data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
                misfit_bw, misfit_sw, stations, best_origin, best_mt, lune_dict)


            print('Saving results...\n')

            # collect information about best-fitting source
            merged_dict = merge_dicts(
                mt_dict,
                lune_dict,
                {'M0': best_mt.moment()},
                {'Mw': best_mt.magnitude()},
                best_origin,
                )

            # save best-fitting source
            # save_json(event_id+'DC+Z_solution.json', merged_dict)


            # save origins
            origins_dict = {_i: origin for _i,origin in enumerate(origins)}
            
            #creating uncertainty plots for all depths
            omega.plot_cdf('cdf/'+str(depths[i])+'.png', results, var=50, nbins=40)
            omega.plot_pdf('pdf/'+str(depths[i])+'.png', results, var= 50, nbins=40)
            omega.plot_rho_vs_V('rho_vs_v/'+str(depths[i])+'.png', results, unc, var=50, nbins=40)
            omega.probability_vs_omega('prob_vs_omega/'+str(depths[i])+'.png', results)
            omega.misfit_vs_omega('misfit_vs_omega/'+str(depths[i])+'.png', results)
            
            results1 = results.reset_index()
            dl = results1.loc[results1[0].argmin()]
            result_All = result_All.append(results)
            min_misfit += [min(results1[0])]
            min_misfit_index += [dl.values[2:8]]
            
    test =  np.array(min_misfit_index)
    plot_misfit_depth(event_id + 'DC+Z_misfit_depth_tradeoffs.png', result_All, unc, test, min_misfit, origins1,
                              show_tradeoffs=True, show_magnitudes=True, title=event_id)

    print('\nFinished\n')

