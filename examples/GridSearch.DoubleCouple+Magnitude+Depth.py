#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_misfit_depth, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # This is the most complicated example. For a much simpler one, see
    # SerialGridSearch.DoubleCouple.py
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
    # We will search over a range of depths about the catalog origin
    #


    catalog_origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        'id': '20090407201255351'
        })

    depths = np.array(
         # depth in meters
        [25000., 30000., 35000., 40000.,                    
         45000., 50000., 55000., 60000.])

    origins = []
    for depth in depths:
        origins += [catalog_origin.copy()]
        setattr(origins[-1], 'depth_in_m', depth)



    #
    # Next, we specify the moment tensor grid and source-time function
    #

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.3, 4.4, 4.5,     
         4.6, 4.7, 4.8]) 

    grid = DoubleCoupleGridRegular(
        npts_per_axis=30,
        magnitudes=magnitudes)

    wavelet = Trapezoid(
        magnitude=4.5)


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

    if comm.rank==0:
        print('Evaluating body wave misfit...\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, grid)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, grid)


    #
    # Generate figures and save results
    #

    if comm.rank==0:

        results = results_bw + results_sw

        # origin corresponding to minimum misfit
        best_origin = origins[results.idxmin('origin')]
        origin_dict = best_origin.as_dict()

        # source corresponding to minimum misfit
        idx = results.idxmin('source')
        best_source = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = grid.get(idx).as_dict()

        components_bw = data_bw.get_components()
        components_sw = data_sw.get_components()

        greens_bw = greens_bw.select(best_origin)
        greens_sw = greens_sw.select(best_origin)

        # synthetics corresponding to minimum misfit
        synthetics_bw = greens_bw.get_synthetics(
            best_source, components_bw, mode='map')

        synthetics_sw = greens_sw.get_synthetics(
            best_source, components_sw, mode='map')

        # time shifts and other attributes corresponding to minimum misfit
        list_bw = misfit_bw.collect_attributes(
            data_bw, greens_bw, best_source)

        list_sw = misfit_sw.collect_attributes(
            data_sw, greens_sw, best_source)

        dict_bw = {station.id: list_bw[_i] 
            for _i,station in enumerate(stations)}

        dict_sw = {station.id: list_sw[_i] 
            for _i,station in enumerate(stations)}


        print('Generating figures...\n')

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, best_origin, best_source, lune_dict)

        plot_misfit_depth(event_id+'_misfit_depth.png',
            results, origins, grid)


        print('Saving results...\n')

        # save best-fitting source
        save_json(event_id+'DC_mt.json', mt_dict)
        save_json(event_id+'DC_lune.json', lune_dict)


        # save time shifts and other attributes
        os.makedirs(event_id+'DC_attrs', exist_ok=True)

        save_json(event_id+'DC_attrs/bw.json', dict_bw)
        save_json(event_id+'DC_attrs/sw.json', dict_sw)


        # save processed waveforms as binary files
        os.makedirs(event_id+'DC_waveforms', exist_ok=True)

        data_bw.write(event_id+'DC_waveforms/dat_bw.p')
        data_sw.write(event_id+'DC_waveforms/dat_sw.p')

        synthetics_bw.write(event_id+'DC_waveforms/syn_bw.p')
        synthetics_sw.write(event_id+'DC_waveforms/syn_sw.p')

        results.save(event_id+'DC_misfit.nc')


        print('\nFinished\n')

