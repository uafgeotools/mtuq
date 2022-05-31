#!/usr/bin/env python

import os
import numpy as np
from mtuq.misfit import Misfit, PolarityMisfit

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune, beachball_pygmt
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit, PolarityMisfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #


    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'


    #
    # For our objective function, we will use a finite count of mismatching
    # polarity orientations
    #

    pmisfit = PolarityMisfit(taup_model=model, polarity_keyword='user3')


    #
    # User-supplied weights control how much each station contributes to the
    # objective function. Weight file can contain polarity information.
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=15,
        magnitudes=[4.4])


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

        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model)

    else:
        stations = None
        data = None
        greens = None

    stations = comm.bcast(stations, root=0)
    data = comm.bcast(data, root=0)
    greens = comm.bcast(greens, root=0)

    #
    # The main computational work starts now
    #

    # Picked polarity values. len(polarity_input) = len(data)
    polarity_input = np.array([-1, -1, -1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 0 ,1, 1, 1, -1, 1, 1, 0])

    # Polulating SAC headers to test mtuq.Database input
    for i in range(len(data)):
        data[i][0].stats.sac['user3'] = polarity_input[i]

    # Polulating SAC headers to test mtuq.GreensTensorList input
    for i in range(len(greens)):
        greens[i][0].stats.sac['user3'] = polarity_input[i]

    if comm.rank==0:
        print('Evaluating polarity misfit...\n')


    # Using an array as input
    results = grid_search(
    polarity_input, greens, pmisfit, origin, grid)

    # Using mtuq.Dataset as input
    results = grid_search(
    data, greens, pmisfit, origin, grid)

    # Using mtuq.GreensTensorList as input
    results = grid_search(
    greens, greens, pmisfit, origin, grid)

    # Using weight file path as input
    # This one relies on appending `/+1` or `/-1` after the station ID in the CAP weight file in `path_weights`.
    # results = grid_search(
    # path_weights, greens, pmisfit, origin, grid)


    if comm.rank==0:

        #
        # Generate figures and save results
        #

        print('Generating figures...\n')

        plot_misfit_lune(event_id+'FMT_misfit.png', results)

        print('Saving results...\n')

        # save misfit surface
        results.save(event_id+'FMT_misfit.nc')


        # This search will have several matching moment tensor.
        # For illustration sake, we pick one of the minimum solution at random
        # from the list of best fitting moment tensor and plot it.
        values = results.values.reshape(len(grid))
        min_indexes = []
        min_value = min(values)
        import random

        # List all minimum values indexes
        for i in range(len(values)):
            if values[i] == min_value:
                min_indexes.append(i)
        # Select one at random
        random_index = random.randrange(len(min_indexes))
        random_value = min_indexes[random_index]
        # Then plot it
        beachball_pygmt('polarity.pdf', polarity_input, greens, grid.get(random_value))

        print('\nFinished\n')
