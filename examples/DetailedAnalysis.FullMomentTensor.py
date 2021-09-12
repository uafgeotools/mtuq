#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune,\
    plot_likelihood_lune, plot_marginal_vw,\
    calculate_likelihoods, calculate_marginals,\
    plot_time_shifts, plot_amplitude_ratios,\
    _plot_lune, _plot_vw, _vw_product
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit.waveform import Misfit, estimate_sigma, calculate_norm_data
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Peforms detailed analysis involving
    #
    # - grid search over all moment tensor parameters except magnitude
    # - separate body wave, Rayleigh wave and Love wave data categories
    # - data variance estimation and likelihood analysis
    #
    #
    # Generates figures of
    #
    # - maximum likelihood surfaces
    # - marginal likelihood surfaces
    # - data misfit surfaces
    # - geographic variation of time shifts between data and synthetics
    # - geographic variation of amplitude ratios between data and synthetics
    #
    #
    # USAGE
    #   mpirun -n <NPROC> python DetailedAnalysis.FullMomentTensor.py
    #   
    #
    # For simpler examples, see SerialGridSearch.DoubleCouple.py and
    # GridSearch.FullMomentTensor.py
    #
    # For ideas on applying this type of analysis to entire sets of events,
    # see github.com/rmodrak/mtbench
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

    misfit_rayleigh = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR'],
        )

    misfit_love = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=15,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        'id': '20090407201255351'
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
        greens = download_greens_tensors(stations, origin, model)

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
        data_bw, greens_bw, misfit_bw, origin, grid)

    if comm.rank==0:
        print('Evaluating Rayleigh wave misfit...\n')

    results_rayleigh = grid_search(
        data_sw, greens_sw, misfit_rayleigh, origin, grid)


    if comm.rank==0:
        print('Evaluating Love wave misfit...\n')

    results_love = grid_search(
        data_sw, greens_sw, misfit_love, origin, grid)

    if comm.rank==0:

        results_sum = results_bw + results_rayleigh + results_love

        #
        # Data variance estimation and likelihood analysis
        #

        # use minimum misfit as initial guess for maximum likelihood
        idx = results_sum.idxmin('source')
        best_source = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = grid.get(idx).as_dict()


        sigma_bw = estimate_sigma(data_bw, greens_bw,
            best_source, misfit_bw.norm, ['Z', 'R'],
            misfit_bw.time_shift_min, misfit_bw.time_shift_max)

        sigma_rayleigh = estimate_sigma(data_sw, greens_sw,
            best_source, misfit_rayleigh.norm, ['Z', 'R'],
            misfit_rayleigh.time_shift_min, misfit_rayleigh.time_shift_max)

        sigma_love = estimate_sigma(data_sw, greens_sw,
            best_source, misfit_love.norm, ['T'],
            misfit_love.time_shift_min, misfit_love.time_shift_max)


        print('  Body wave variance:  %.3e' %
            sigma_bw**2)
        print('  Rayleigh variance:   %.3e' %
            sigma_rayleigh**2)
        print('  Love variance:       %.3e' %
            sigma_love**2)

        print('\n')


        # maximum likelihood surface
        likelihoods = _vw_product(
            calculate_likelihoods(results_bw, sigma_bw**2),
            calculate_likelihoods(results_rayleigh, sigma_rayleigh**2),
            calculate_likelihoods(results_love, sigma_love**2))

        # marginal likelihood surface
        marginals = _vw_product(
            calculate_marginals(results_bw, sigma_bw**2),
            calculate_marginals(results_rayleigh, sigma_rayleigh**2),
            calculate_marginals(results_love, sigma_love**2))

        #
        # Generate figures and save results
        #

        # only generate components present in the data
        components_bw = data_bw.get_components()
        components_sw = data_sw.get_components()

        # synthetics corresponding to minimum misfit
        synthetics_bw = greens_bw.get_synthetics(
            best_source, components_bw, mode='map')

        synthetics_sw = greens_sw.get_synthetics(
            best_source, components_sw, mode='map')


        # time shifts and other attributes corresponding to minimum misfit
        list_bw = misfit_bw.collect_attributes(
            data_bw, greens_bw, best_source)

        list_rayleigh = misfit_rayleigh.collect_attributes(
            data_sw, greens_sw, best_source)

        list_love = misfit_love.collect_attributes(
            data_sw, greens_sw, best_source)

        list_sw = [{**list_rayleigh[_i], **list_love[_i]}
            for _i in range(len(stations))]

        dict_bw = {station.id: list_bw[_i] 
            for _i,station in enumerate(stations)}

        dict_rayleigh = {station.id: list_rayleigh[_i] 
            for _i,station in enumerate(stations)}

        dict_love = {station.id: list_love[_i] 
            for _i,station in enumerate(stations)}

        dict_sw = {station.id: list_sw[_i] 
            for _i,station in enumerate(stations)}



        print('Plotting observed and synthetic waveforms...\n')

        plot_beachball(event_id+'FMT_beachball.png', best_source)

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
            misfit_bw, misfit_rayleigh, stations, origin, best_source, lune_dict)


        print('Plotting misfit surfaces...\n')

        os.makedirs(event_id+'FMT_misfit', exist_ok=True)

        plot_misfit_lune(event_id+'FMT_misfit/bw.png', results_bw,
            title='Body wave misfit (%s)' % misfit_bw.norm)

        plot_misfit_lune(event_id+'FMT_misfit/sw.png', results_rayleigh,
            title='Rayleigh wave misfit (%s)' % misfit_rayleigh.norm)

        plot_misfit_lune(event_id+'FMT_misfit/love.png', results_love,
            title='Love wave misfit (%s)' % misfit_love.norm)

        print()


        print('Plotting maximum likelihood surfaces...\n')

        os.makedirs(event_id+'FMT_likelihood', exist_ok=True)

        plot_likelihood_lune(event_id+'FMT_likelihood/bw.png',
            results_bw, var=sigma_bw**2, 
            title='Body waves')

        plot_likelihood_lune(event_id+'FMT_likelihood/rayleigh.png',
            results_rayleigh, var=sigma_rayleigh**2, 
            title='Rayleigh waves')

        plot_likelihood_lune(event_id+'FMT_likelihood/love.png',
            results_love, var=sigma_love**2, 
            title='Love waves')

        _plot_lune(event_id+'FMT_likelihood/all.png',
            likelihoods, colormap='hot_r',
            title='All data categories')

        print()


        print('Plotting marginal likelihood surfaces...\n')

        os.makedirs(event_id+'FMT_marginal', exist_ok=True)

        plot_marginal_vw(event_id+'FMT_marginal/bw.png',
            results_bw, var=sigma_bw**2,
            title='Body waves')

        plot_marginal_vw(event_id+'FMT_marginal/rayleigh.png',
            results_rayleigh, var=sigma_rayleigh**2,
            title='Rayleigh waves')

        plot_marginal_vw(event_id+'FMT_marginal/love.png',
            results_love, var=sigma_love**2,
            title='Love waves')

        #_plot_vw(event_id+'FMT_marginal/all.png',
        #    marginals, colormap='hot_r',
        #    title='All data categories')


        print('Plotting time shift geographic variation...\n')

        plot_time_shifts(event_id+'FMT_time_shifts/bw',
            list_bw, stations, origin, best_source)

        plot_time_shifts(event_id+'FMT_time_shifts/sw',
            list_sw, stations, origin, best_source)


        print('Plotting amplitude ratio geographic variation...\n')

        plot_amplitude_ratios(event_id+'FMT_amplitude_ratios/bw',
            list_bw, stations, origin, best_source)

        plot_amplitude_ratios(event_id+'FMT_amplitude_ratios/sw',
            list_sw, stations, origin, best_source)


        print('\nSaving results...\n')

        # save best-fitting source
        save_json(event_id+'FMT_mt.json', mt_dict)
        save_json(event_id+'FMT_lune.json', lune_dict)


        # save time shifts and other attributes
        os.makedirs(event_id+'FMT_attrs', exist_ok=True)

        save_json(event_id+'FMT_attrs/bw.json', dict_bw)
        save_json(event_id+'FMT_attrs/sw.json', dict_sw)


        # save processed waveforms as binary files
        os.makedirs(event_id+'FMT_waveforms', exist_ok=True)

        data_bw.write(event_id+'FMT_waveforms/dat_bw.p')
        data_sw.write(event_id+'FMT_waveforms/dat_sw.p')

        synthetics_bw.write(event_id+'FMT_waveforms/syn_bw.p')
        synthetics_sw.write(event_id+'FMT_waveforms/syn_sw.p')


        # save misfit surfaces as netCDF files
        results_bw.save(event_id+'FMT_misfit/bw.nc')
        results_rayleigh.save(event_id+'FMT_misfit/rayleigh.nc')
        results_love.save(event_id+'FMT_misfit/love.nc')
        results_sum.save(event_id+'FMT_misfit/sum.nc')


        print('\nFinished\n')

