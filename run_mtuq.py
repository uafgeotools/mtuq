import os
import sys

import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_misfit_depth, plot_misfit_dc, plot_variance_reduction_lune
from mtuq.graphics.uq.origin_depth import _get_depths, _get_sources, _min_dataarray
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.misfit.waveform import calculate_norm_data
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

from mpi4py import MPI

# Tester script just trying to get the information I need
# Event is the July 2021 event that capuaf has been run for

if len(sys.argv) != 2:
    raise Exception('proper useage: python run_mtuq.py eid')
else:
    eid = sys.argv[1]
    eid = str(eid)
    print('event id: ', eid)


path_data = fullpath('%s/*.[zrt]' % eid)
path_weights = fullpath('%s/weight.dat' % eid)
#model = 'ak135'

db = open_db('/store/wf/FK_synthetics/tactmod',format='FK')
model='tactmod'

# Process the body and surface waves separately

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min= 0.25,
    freq_max= 0.6667,
    pick_type='FK_metadata',
    FK_database='/store/wf/FK_synthetics/tactmod',
    window_type='body_wave',
    window_length=15.,
    capuaf_file=path_weights,
)

process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.0333,
    freq_max=0.0625,
    pick_type='FK_metadata',
    FK_database='/store/wf/FK_synthetics/tactmod',
    window_type='surface_wave',
    window_length=120.,
    capuaf_file=path_weights,
)

# Objective function

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

misfit_sw = Misfit(
    norm='L2',
    time_shift_min=-10.,
    time_shift_max=+10.,
    time_shift_groups=['ZR','T'],
)

# Weight file
station_id_list = parse_station_codes(path_weights)

# Begin setup for depth search
catalog_origin = Origin({
    'time': '2021-07-23T04:20:20.018000Z',
    'latitude': 64.461,
    'longitude': -146.850,
    'depth_in_m': 11700.000,
    'id': str(eid)
})

# Depths (in meters) to search over
depths = np.array([1000., 2000., 4000., 5000., 6000., 7000., 8000., 9000., 10000., 12000., 13000., 14000.]) #tactmod has boundary at 3,11km

origins = []
for depth in depths:
    origins += [catalog_origin.copy()]
    setattr(origins[-1], 'depth_in_m', depth)

# Magnitudes to search over
magnitudes = np.array([4.40, 4.45, 4.50, 4.55])

grid = DoubleCoupleGridRegular(
    npts_per_axis=30,
    magnitudes=magnitudes)

# Define source-time function
wavelet = Trapezoid(magnitude=4.7)

comm = MPI.COMM_WORLD


# Main I/O starts here


if comm.rank==0:
    print('Reading data...\n')
    data = read(path_data, format='sac', 
        event_id=eid,
        station_id_list=station_id_list,
        tags=['units:cm', 'type:velocity']) 


    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    data_all = data_bw + data_sw


    print('Reading Greens functions...\n')
    #greens = download_greens_tensors(stations, origins, model)
    greens = db.get_greens_tensors(stations, origins)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)
    greens_all = greens_bw + greens_sw


else:
    stations = None
    data_bw = None
    data_sw = None
    greens_bw = None
    greens_sw = None


stations = comm.bcast(stations, root=0)
data_all = comm.bcast(data, root=0)
data_bw = comm.bcast(data_bw, root=0)
data_sw = comm.bcast(data_sw, root=0)
greens_bw = comm.bcast(greens_bw, root=0)
greens_sw = comm.bcast(greens_sw, root=0)

# Main computational work
if comm.rank==0:
    print('Evaluating body wave misfit...\n')

results_bw = grid_search(
    data_bw, greens_bw, misfit_bw, origins, grid)

if comm.rank==0:
    print('Evaluating rayleigh wave misfit...\n')

results_rayleigh = grid_search(
    data_sw, greens_sw, misfit_rayleigh, origins, grid)

if comm.rank==0:
    print('Evaluating love wave misfit...\n')

results_love     = grid_search(
    data_sw, greens_sw, misfit_love, origins, grid)

# Generate figures and save results
if comm.rank == 0:
    results = results_bw + results_rayleigh + results_love
    
    print(results)

    
    # Source corresponding to minimum misfit
    idx = results.idxmin('source')
    best_source = grid.get(idx)
    lune_dict = grid.get_dict(idx)
    mt_dict = grid.get(idx).as_dict()

    # Calculate norms for variance reduction
    norm_bw = calculate_norm_data(data_bw, 
        misfit_bw.norm, ['Z', 'R'])
    norm_rayleigh = calculate_norm_data(data_sw, 
        misfit_rayleigh.norm, ['Z', 'R'])
    norm_love = calculate_norm_data(data_sw, 
        misfit_love.norm, ['T'])

    data_norm = norm_bw + norm_rayleigh + norm_love
    norms = {misfit_bw.norm+'_bw': norm_bw,
             misfit_rayleigh.norm+'_rayleigh': norm_rayleigh,
             misfit_love.norm+'_love': norm_love,
             'L2__data_norm': data_norm}
    save_json(eid+'_data_norms.json',norms)

    # origin corresponding to minimum misfit
    best_origin = origins[results.idxmin('origin')]
    origin_dict = best_origin.as_dict()

    # Best sources for each depth
    depths = _get_depths(origins)
    values, indices = _min_dataarray(results)
    best_sources = _get_sources(grid,indices)
    source_depth = {}
    for ii in range(len(best_sources)):
        temp = best_sources[ii].as_dict()
        temp['misfit'] = values[ii]
        vr_temp = 100*(1-values[ii]/data_norm)
        temp['vr'] = vr_temp
        km = int(depths[ii]/1000)
        source_depth[km] = temp
    
    save_json(eid+'_depths_source.json',source_depth)
    # only generate components present in the data
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

    print('Generating figures...\n')

    plot_data_greens2(eid+'DC+_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, best_origin, best_source, lune_dict)

    plot_misfit_depth(eid+'DC+_misfit_depth.png',
        results, origins, grid)

    plot_variance_reduction_lune(eid+'DC_vr_bw.png',
        results_bw, norm_bw, title='Body waves',
        colorbar_label='Variance reduction (percent)')

    plot_variance_reduction_lune(eid+'DC_vr_rayleigh.png',
        results_rayleigh, norm_rayleigh, title='Rayleigh waves',
        colorbar_label='Variance reduction (percent)')

    plot_variance_reduction_lune(eid+'DC_vr_love.png',
        results_love, norm_love, title='Love waves', 
        colorbar_label='Variance reduction (percent)')

    plot_variance_reduction_lune(eid+'DC_vr_data.png',results,data_norm,title='data norm',colorbar_label='Variance reduction (percent)')
    
    # save stations and origins
    stations_dict = {station.id: station
        for _i,station in enumerate(stations)}

    save_json(eid+'DC_stations.json', stations_dict)
    save_json(eid+'DC_origin.json', {0: catalog_origin})

    # save best fit origin
    save_json(eid+'DC_best_origin.json', origin_dict)

    # save best fit source
    save_json(eid+'DC_mt.json', mt_dict)
    save_json(eid+'DC_lune.json',lune_dict)

    # save time shifts and amplitude ratios
    save_json(eid+'_bw.json', dict_bw)
    save_json(eid+'_rayleigh.json', dict_rayleigh)
    save_json(eid+'_love.json', dict_love)
    
    #save misfit values
    results.save(eid+'misfit.nc')
