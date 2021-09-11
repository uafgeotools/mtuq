

Imports="""
import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid


"""


Docstring_GridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double-couple moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple.py
    #
    # For a simpler example, see SerialGridSearch.DoubleCouple.py, 
    # which runs the same inversion in serial
    #

"""


Docstring_GridSearch_DoubleCoupleMagnitudeDepth="""
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

"""


Docstring_GridSearch_FullMomentTensor="""
if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters except
    # magnitude 
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

"""


Docstring_SerialGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double-couple moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 60 seconds. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #

"""


Docstring_TestGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK:
    #
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351
    #
    # Note however that CAP uses a different method for defining regular grids
    #

"""


Docstring_TestGridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we included mangitude and depth and use a coarser grid
    #
"""


Docstring_TestGraphics="""
if __name__=='__main__':
    #
    # Tests data, synthetics and beachball plotting utilities
    #
    # Note that in the figures created by this script, the data and synthetics 
    # are not expected to fit epsecially well; currently, the only requirement 
    # is that the script runs without errors
    #

    import matplotlib
    matplotlib.use('Agg', warn=False, force=True)
    import matplotlib
"""



Docstring_TestMisfit="""
if __name__=='__main__':
    #
    # Checks the correctness of the fast (optimized) misfit function
    # implementations against a simple pure Python implementation.
    # These implementations correspond to:
    #
    #   optimization_level=0: simple pure Python
    #   optimization_level=1: fast pure Python
    #   optimization_level=2: fast Python/C
    #
    # In running the test in our environment, we observe that the two pure 
    # Python implementations agree almost exactly.  On the other hand, the
    # pure Python and Python/C results differ by as much as 0.1 percent, 
    # presumably as a result of differences in the way that floating-point
    # error accumulates in the sum over residuals. Further work is required to 
    # understand this better
    #
    # Possibly relevant is the fact that C extensions are compiled with
    # `-Ofast` flag, as specified in `setup.py`.
    #
    # Note that the `optimization_level` keyword argument does not correspond
    # at all to C compiler optimization flags.  For example, the NumPy binaries
    # called by the simple pure Python misfit function are probably compiled 
    # using a nonzero optimization level?
    #


"""


Docstring_Gallery="""
if True:
    #
    # Creates example data structures
    #
    # Rather than being executed as a script, this code is designed to be
    # imported.  After importing this module, users can access the example data
    # and functions listed in __all__
    #
    # Note that some I/O and data processing are involved in creating the
    # example data, so importing this module may significantly longer than
    # other modules
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
"""


Docstring_BenchmarkCAP="""
if __name__=='__main__':
    #
    # Given seven "fundamental" moment tensors, generates MTUQ synthetics and
    # compares with corresponding CAP/FK synthetics
    #
    # Before running this script, it is necessary to unpack the CAP/FK 
    # synthetics using data/tests/unpack.bash
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we consider only seven grid points rather than an entire
    # grid, and here the final plots are a comparison of MTUQ and CAP/FK 
    # synthetics rather than a comparison of data and synthetics
    #
    # Because of the idiosyncratic way CAP implements source-time function
    # convolution, it's not expected that CAP and MTUQ synthetics will match 
    # exactly. CAP's "conv" function results in systematic magnitude-
    # dependent shifts between origin times and arrival times. We deal with 
    # this by applying magnitude-dependent time-shifts to MTUQ synthetics 
    # (which normally lack such shifts) at the end of the benchmark. Even with
    # this correction, the match will not be exact because CAP applies the 
    # shifts before tapering and MTUQ after tapering. The resulting mismatch 
    # will usually be apparent in body-wave windows, but not in surface-wave 
    # windows
    #
    # Note that CAP works with dyne,cm and MTUQ works with N,m, so to make
    # comparisons we convert CAP output from the former to the latter
    #
    # The CAP/FK synthetics used in the comparison were generated by 
    # uafseismo/capuaf:46dd46bdc06e1336c3c4ccf4f99368fe99019c88
    # using the following commands
    #
    # source #0 (explosion):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/1.178/90/45/90 20090407201255351
    #
    # source #1 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/90/45/90 20090407201255351
    #
    # source #2 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/180 20090407201255351
    #
    # source #3 (on-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/0 20090407201255351
    #
    # source #4 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/90/90 20090407201255351
    #
    # source #5 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/0/0 20090407201255351
    #
    # source #6 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/0/90/180 20090407201255351
    #

"""


ArgparseDefinitions="""
    # by default, the script runs with figure generation and error checking
    # turned on
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)

"""


Paths_BenchmarkCAP="""
    from mtuq.util.cap import\\
        get_synthetics_cap, get_synthetics_mtuq,\\
        get_data_cap, compare_cap_mtuq


    # the following directories correspond to the moment tensors in the list 
    # "sources" below
    paths = []
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/0')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/1')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/2')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/3')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/4')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/5')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/6')]

"""


PathsComments="""
    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #
"""


Paths_Syngine="""
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'

"""


Paths_AxiSEM="""
    path_greens= '/home/rmodrak/data/ak135f_scak-2s'
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135f_scak-2s'

"""


Paths_FK="""
    path_greens=  fullpath('data/tests/benchmark_cap/greens/scak')
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'scak'

"""



DataProcessingComments="""
    #
    # Body and surface wave measurements will be made separately
    #
"""


DataProcessingDefinitions="""
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

"""


MisfitComments="""
    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #
"""


MisfitDefinitions="""
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

"""



WeightsComments="""
    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
"""


WeightsDefinitions="""
    station_id_list = parse_station_codes(path_weights)

"""


OriginComments="""
    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #
"""


OriginDefinitions="""
    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        'id': '20090407201255351'
        })

"""


OriginsComments="""
    #
    # We will search over a range of depths about the catalog origin
    #

"""


OriginsDefinitions="""
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


"""


Grid_DoubleCouple="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=40,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_DoubleCoupleMagnitudeDepth="""
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

"""


Grid_FullMomentTensor="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=15,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_TestDoubleCoupleMagnitudeDepth="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitudes=[4.4, 4.5, 4.6])

    wavelet = Trapezoid(
        magnitude=4.5)

"""+OriginDefinitions+"""
    depths = np.array(
         # depth in meters
        [34000])

    origins = []
    for depth in depths:
        origin.depth = depth
        origins += [origin.copy()]
"""


Grid_TestGraphics="""
    mt = np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]) # explosion
    mt *= 1.e16

    wavelet = Trapezoid(
        magnitude=4.5)
"""


Grid_BenchmarkCAP="""
    #
    # Next we specify the source parameter grid
    #

    magnitude = 4.5
    moment = 10.**(1.5*magnitude + 9.1) # units: N-m

    sources = []
    for array in [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (on-diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (on-diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (on-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 1., 0., 0.]), # source 4 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 1., 0.]), # source 5 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 0., 1.]), # source 6 (off-diagonal)
       ]:

        sources += [MomentTensor(np.sqrt(2)*moment*array)]

    wavelet = Trapezoid(
        magnitude=magnitude)

"""


Main_GridSearch="""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:cm', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()


        print('Processing data...\\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\\n')
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\\n')
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
        print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, grid)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)

"""


Main1_SerialGridSearch_DoubleCouple="""
    #
    # The main I/O work starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:cm', 'type:velocity']) 


    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)


    print('Reading Greens functions...\\n')
    greens = download_greens_tensors(stations, origin, model)


    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)

"""


Main2_SerialGridSearch_DoubleCouple="""
    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\\n')
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)

    print('Evaluating surface wave misfit...\\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)

"""


Main_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main I/O work starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac', 
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)


    print('Reading Greens functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origins)


    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, grid, 0)

    print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, grid, 0)


"""


Main_TestGraphics="""

    print('Reading data...\\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:cm', 'type:velocity'])


    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Greens functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # Generate figures
    #

    print('Figure 1 of 3\\n')

    plot_data_greens2('graphics_test_1.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, header=False)

    print('Figure 2 of 3\\n')

    plot_data_greens2('graphics_test_2.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, header=False)

    print('Figure 3 of 3\\n')

    plot_beachball('graphics_test_3.png', mt)

    print('\\nFinished\\n')
"""



Main_TestMisfit="""
    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\\n')

    results_0 = misfit_bw(
        data_bw, greens_bw, grid, optimization_level=0)

    results_1 = misfit_bw(
        data_bw, greens_bw, grid, optimization_level=1)

    results_2 = misfit_bw(
        data_bw, greens_bw, grid, optimization_level=2)

    print('  optimization level:  0\\n', 
          '  argmin:  %d\\n' % results_0.argmin(), 
          '  min:     %e\\n\\n' % results_0.min())

    print('  optimization level:  1\\n', 
          '  argmin:  %d\\n' % results_1.argmin(), 
          '  min:     %e\\n\\n' % results_1.min())

    print('  optimization level:  2\\n', 
          '  argmin:  %d\\n' % results_2.argmin(), 
          '  min:     %e\\n\\n' % results_2.min())

    print('')

    assert results_0.argmin()==results_1.argmin()==results_2.argmin()


    print('Evaluating surface wave misfit...\\n')

    results_0 = misfit_sw(
        data_sw, greens_sw, grid, optimization_level=0)

    results_1 = misfit_sw(
        data_sw, greens_sw, grid, optimization_level=1)

    results_2 = misfit_sw(
        data_sw, greens_sw, grid, optimization_level=2)

    print('  optimization level:  0\\n', 
          '  argmin:  %d\\n' % results_0.argmin(), 
          '  min:     %e\\n\\n' % results_0.min())

    print('  optimization level:  1\\n', 
          '  argmin:  %d\\n' % results_1.argmin(), 
          '  min:     %e\\n\\n' % results_1.min())

    print('  optimization level:  2\\n', 
          '  argmin:  %d\\n' % results_2.argmin(), 
          '  min:     %e\\n\\n' % results_2.min())

    assert results_0.argmin()==results_1.argmin()==results_2.argmin()


"""


WrapUp_DetailedAnalysis_FullMomentTensor="""
    #
    # Analyzing results
    #

    if comm.rank==0:

        results = results_bw + results_sw

        # source corresponding to minimum misfit
        idx = results.idxmin('source')
        best_source = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = grid.get(idx).as_dict()

        components_bw = data_bw.get_components()
        components_sw = data_sw.get_components()

        # synthetics corresponding to minimum misfit
        synthetics_bw = greens_bw.get_synthetics(
            best_source, components_bw, mode='map')

        synthetics_sw = greens_sw.get_synthetics(
            best_source, components_sw, mode='map')

        # time shifts corresponding to minimum misfit
        attrs_bw = misfit_bw.collect_attributes(
            data_bw, greens_bw, best_source)

        attrs_sw = misfit_sw.collect_attributes(
            data_sw, greens_sw, best_source)


        print('Generating figures...\\n')

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

        plot_beachball(event_id+'FMT_beachball.png', best_source)

        plot_misfit_lune(event_id+'DC_misfit.png', results)


        print('Saving results...\\n')

        os.makedirs(event_id+'FMT/waveforms/data', exist_ok=True)
        os.makedirs(event_id+'FMT/waveforms/synthetics', exist_ok=True)

        os.makedirs(event_id+'FMT/misfit/bw', exist_ok=True)
        os.makedirs(event_id+'FMT/misfit/sw', exist_ok=True)

        save_json(event_id+'FMT_mt.json', mt_dict)
        save_json(event_id+'FMT_lune.json', lune_dict)

        data_bw.write(event_id+'FMT/waveforms/data/bw.p')
        data_sw.write(event_id+'FMT/waveforms/data/sw.p')

        synthetics_bw.write(event_id+'FMT/waveforms/synthetics/bw.p')
        synthetics_sw.write(event_id+'FMT/waveforms/synthetics/sw.p')

        results_bw.save(event_id+'FMT/misfit/bw.nc')
        results_sw.save(event_id+'FMT/misfit/sw.nc')

        for _i, station in enumerate(stations):
            for key in attrs_bw[_i]:
                filename = event_id+'FMT/misfit/bw/'+station.id+key
                save_json(filename, attrs_bw[_i][key])

            for key in attrs_sw[_i]:
                filename = event_id+'FMT/misfit/sw/'+station.id+key
                save_json(filename, attrs_sw[_i][key])

        print('\\nFinished\\n')
"""




WrapUp_GridSearch="""
    #
    # Analyzing results
    #

    if comm.rank==0:

        results = results_bw + results_sw

        # source corresponding to minimum misfit
        idx = results.idxmin('source')
        best_source = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = grid.get(idx).as_dict()

        components_bw = data_bw.get_components()
        components_sw = data_sw.get_components()

        # synthetics corresponding to minimum misfit
        synthetics_bw = greens_bw.get_synthetics(
            best_source, components_bw, mode='map')

        synthetics_sw = greens_sw.get_synthetics(
            best_source, components_sw, mode='map')


        print('Generating figures...\\n')

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

        plot_beachball(event_id+'DC_beachball.png', best_source)

        plot_misfit_dc(event_id+'DC_misfit.png', results)


        print('Saving results...\\n')

        save_json(event_id+'DC_mt.json', mt_dict)
        save_json(event_id+'DC_lune.json', lune_dict)

        os.makedirs(event_id+'DC_waveforms/data', exist_ok=True)
        data_bw.write(event_id+'DC_waveforms/data/bw.p')
        data_sw.write(event_id+'DC_waveforms/data/sw.p')

        os.makedirs(event_id+'DC_waveforms/synthetics', exist_ok=True)
        synthetics_bw.write(event_id+'DC_waveforms/synthetics/bw.p')
        synthetics_sw.write(event_id+'DC_waveforms/synthetics/sw.p')

        results.save(event_id+'DC.nc')


        print('\\nFinished\\n')
"""


WrapUp_GridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # Analyzing results
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


        print('Generating figures...\\n')

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

        plot_misfit_depth(event_id+'_misfit_depth.png',
            results, origins, grid)


        print('Saving results...\\n')

        save_json(event_id+'DC_mt.json', mt_dict)
        save_json(event_id+'DC_lune.json', lune_dict)
        save_json(event_id+'DC_origin.json', origin_dict)

        os.makedirs(event_id+'DC_waveforms/data', exist_ok=True)
        data_bw.write(event_id+'DC_waveforms/data/bw.p')
        data_sw.write(event_id+'DC_waveforms/data/sw.p')

        os.makedirs(event_id+'DC_waveforms/synthetics', exist_ok=True)
        synthetics_bw.write(event_id+'DC_waveforms/synthetics/bw.p')
        synthetics_sw.write(event_id+'DC_waveforms/synthetics/sw.p')

        results.save(event_id+'DC.nc')


        print('\\nFinished\\n')
"""


WrapUp_SerialGridSearch_DoubleCouple="""
    #
    # Analyzing results
    #

    results = results_bw + results_sw

    # source corresponding to minimum misfit
    idx = results.idxmin('source')
    best_source = grid.get(idx)
    lune_dict = grid.get_dict(idx)
    mt_dict = grid.get(idx).as_dict()

    components_bw = data_bw.get_components()
    components_sw = data_sw.get_components()

    # synthetics corresponding to minimum misfit
    synthetics_bw = greens_bw.get_synthetics(
        best_source, components_bw, mode='map')

    synthetics_sw = greens_sw.get_synthetics(
        best_source, components_sw, mode='map')


    print('Generating figures...\\n')

    plot_data_greens2(event_id+'DC_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

    plot_beachball(event_id+'DC_beachball.png', best_source)

    plot_misfit_dc(event_id+'DC_misfit.png', results)


    print('Saving results...\\n')

    save_json(event_id+'DC_mt.json', mt_dict)
    save_json(event_id+'DC_lune.json', lune_dict)

    os.makedirs(event_id+'DC_waveforms/data', exist_ok=True)
    data_bw.write(event_id+'DC_waveforms/data/bw.p')
    data_sw.write(event_id+'DC_waveforms/data/sw.p')

    os.makedirs(event_id+'DC_waveforms/synthetics', exist_ok=True)
    synthetics_bw.write(event_id+'DC_waveforms/synthetics/bw.p')
    synthetics_sw.write(event_id+'DC_waveforms/synthetics/sw.p')

    results.save(event_id+'DC.nc')


    print('\\nFinished\\n')
"""


WrapUp_TestGridSearch_DoubleCouple="""

    results = results_bw + results_sw

    # source corresponding to minimum misfit
    idx = results.idxmin('source')
    best_source = grid.get(idx)

    lune_dict = grid.get_dict(idx)
    mt_dict = grid.get(idx).as_dict()

    if run_figures:

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

        plot_beachball(event_id+'DC_beachball.png', best_source)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-6):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            for _a, _b, _bool in zip(
                a, b, np.isclose(a, b, atol=atol, rtol=rtol)):

                print('%s:  %.e <= %.1e + %.1e * %.1e' %\\
                    ('passed' if _bool else 'failed', abs(_a-_b), atol, rtol, abs(_b)))
            print('')

            return np.all(
                np.isclose(a, b, atol=atol, rtol=rtol))

        if not isclose(best_source.as_vector(),
            np.array([
                 -6.731618e+15,
                  8.398708e+14,
                  5.891747e+15,
                 -1.318056e+15,
                  7.911756e+14,
                  2.718294e+15,
                 ])
            ):
            raise Exception(
                "Grid search result differs from previous mtuq result")

        print('SUCCESS\\n')
"""


WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    results = results_bw + results_sw

    idx = results.idxmin('source')
    best_source = grid.get(idx)

    if run_figures:
        filename = event_id+'_misfit_vs_depth.png'
        #misfit_vs_depth(filename, best_misfit)

    if run_checks:
        pass

    print('SUCCESS\\n')

"""


Main_BenchmarkCAP="""
    #
    # The benchmark starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac', 
        event_id=event_id,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Greens functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    depth = int(origin.depth_in_m/1000.)+1
    name = '_'.join([model, str(depth), event_id])


    print('Comparing waveforms...')

    for _i, mt in enumerate(sources):
        print('  %d of %d' % (_i+1, len(sources)))

        cap_bw, cap_sw = get_synthetics_cap(
            data_bw, data_sw, paths[_i], name)

        mtuq_bw, mtuq_sw = get_synthetics_mtuq(
            data_bw, data_sw, greens_bw, greens_sw, mt)

        if run_figures:
            plot_waveforms2('cap_vs_mtuq_'+str(_i)+'.png',
                cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
                stations, origin, trace_labels=False)

        if run_checks:
            compare_cap_mtuq(
                cap_bw, cap_sw, mtuq_bw, mtuq_sw)

    if run_figures:
        # "bonus" figure comparing how CAP processes observed data with how
        # MTUQ processes observed data
        mtuq_sw, mtuq_bw = data_bw, data_sw

        cap_sw, cap_bw = get_data_cap(
            data_bw, data_sw, paths[0], name)

        plot_waveforms2('cap_vs_mtuq_data.png',
            cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
            stations, origin, trace_labels=False, normalize=False)

    print('\\nSUCCESS\\n')

"""


if __name__=='__main__':
    import os
    from mtuq.util import basepath, replace
    os.chdir(basepath())


    with open('examples/DetailedAnalysis.FullMomentTensor.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRegular',
            'FullMomentTensorGridSemiregular',
            'plot_misfit_dc',
            'plot_misfit_lune',
            ))
        file.write(Docstring_GridSearch_FullMomentTensor)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main_GridSearch)
        file.write(WrapUp_DetailedAnalysis_FullMomentTensor)


    with open('examples/GridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_GridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main_GridSearch)
        file.write(WrapUp_GridSearch)


    with open('examples/GridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'plot_beachball',
            'plot_misfit_depth',
            ))
        file.write(Docstring_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(OriginsComments)
        file.write(OriginsDefinitions)
        file.write(Grid_DoubleCoupleMagnitudeDepth)
        file.write(
            replace(
            Main_GridSearch,
            'origin',
            'origins',
            ))
        file.write(WrapUp_GridSearch_DoubleCoupleMagnitudeDepth)


    with open('examples/GridSearch.FullMomentTensor.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRegular',
            'FullMomentTensorGridSemiregular',
            'plot_misfit_dc',
            'plot_misfit_lune',
            ))
        file.write(Docstring_GridSearch_FullMomentTensor)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main_GridSearch)
        file.write(
            replace(
            WrapUp_GridSearch,
            'DC',
            'FMT',
            'plot_misfit_dc',
            'plot_misfit_lune',
            ))


    with open('examples/SerialGridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_SerialGridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main1_SerialGridSearch_DoubleCouple)
        file.write(Main2_SerialGridSearch_DoubleCouple)
        file.write(WrapUp_SerialGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_TestGridSearch_DoubleCouple)
        file.write(ArgparseDefinitions)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'npts.*,',
            'npts_per_axis=5,',
            ))
        file.write(WeightsDefinitions)
        file.write(OriginDefinitions)
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
            'greens = download_greens_tensors\(stations, origin, model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n    '
           +'greens = db.get_greens_tensors(stations, origin)',
            ))
        file.write(
            replace(
            Main2_SerialGridSearch_DoubleCouple,
            'origin, grid',
            'origin, grid, 0',
            ))
        file.write(WrapUp_TestGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt_depth.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'plot_beachball',
            'plot_misfit_depth',
            ))
        file.write(Docstring_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(ArgparseDefinitions)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(WeightsDefinitions)
        file.write(Grid_TestDoubleCoupleMagnitudeDepth)
        file.write(Main_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth)


    with open('tests/test_misfit.py', 'w') as file:
        file.write(
            replace(
            Imports,
            ))
        file.write(Docstring_TestMisfit)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'npts.*,',
            'npts_per_axis=5,',
            ))
        file.write(OriginDefinitions)
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
            'greens = download_greens_tensors\(stations, origin, model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n    '
           +'greens = db.get_greens_tensors(stations, origin)',
            ))
        file.write(Main_TestMisfit)


    with open('tests/benchmark_cap_vs_mtuq.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'Origin',
            'MomentTensor',
            'syngine',
            'fk',
            'plot_data_greens2',
            'plot_waveforms2',
            ))
        file.write(Docstring_BenchmarkCAP)
        file.write(ArgparseDefinitions)
        file.write(Paths_BenchmarkCAP)
        file.write(
            replace(
            Paths_FK,
            'data/examples/20090407201255351/weights.dat',
            'data/tests/benchmark_cap/20090407201255351/weights.dat',
            ))
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(
            replace(
            MisfitDefinitions,
            'time_shift_max=.*',
            'time_shift_max=0.,',
            ))
        file.write(Grid_BenchmarkCAP)
        file.write(Main_BenchmarkCAP)


    with open('tests/test_graphics.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_TestGraphics)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_TestGraphics)
        file.write(Main_TestGraphics)


    with open('mtuq/util/gallery.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_Gallery)
        file.write(Paths_Syngine)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'npts.*',
            'npts_per_axis=10,',
            ))
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
            'print.*',
            '',
            ))


