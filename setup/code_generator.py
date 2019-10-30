

Imports="""
import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.graphics import plot_data_greens, plot_beachball
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import Trapezoid


"""


Docstring_GridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
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
    # Double-couple inversion example
    #   
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # This is the most complicated example. For much simpler one, see
    # SerialGridSearch.DoubleCouple.py
    #   

"""


Docstring_GridSearch_FullMomentTensor="""
if __name__=='__main__':
    #
    # Full moment tensor inversion example
    #   
    # Carries out grid search over all moment tensor parameters except
    # magnitude 
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

"""


Docstring_ChinookGridSearch_DoubleCouple="""
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
    #   mpirun -n <NPROC> python ChinookGridSearch.DoubleCouple.py
    #

"""


Docstring_ChinookGridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # THIS EXAMPLE ONLY WORKS ON CHINOOK.ALASKA.EDU
    #

    #
    # CAP-style double-couple inversion example
    # 

    #
    # Carries out grid search over source orientation, magnitude, and depth
    # using Green's functions and phase picks from a local FK database
    #

    #
    # USAGE
    #   mpirun -n <NPROC> python ChinookGridSearch.DoubleCouple+Magntidue+Depth.py
    #

"""


Docstring_SerialGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 10 minutes. For faster results try 
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
    # Tests misfit functions
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
    # "grid" below
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
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake
    #
"""


Paths_Syngine="""
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135'

"""


Paths_AxiSEM="""
    path_greens= '/home/rmodrak/data/ak135f_scak-2s'
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135f_scak-2s'

"""


Paths_FK="""
    path_greens=  fullpath('data/tests/benchmark_cap/greens/scak')
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'

"""



DataProcessingComments="""
    #
    # Body- and surface-wave data are processed separately and held separately 
    # in memory
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
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #
"""


MisfitDefinitions="""
    misfit_bw = Misfit(
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )

"""


Grid_DoubleCouple="""
    #
    # Following obspy, we use the variable name "source" for the mechanism of
    # an event and "origin" for the location of an event
    #

    sources = DoubleCoupleGridRandom(
        npts=50000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_DoubleCoupleMagnitudeDepth="""
    #
    # Following obspy, we use the variable name "source" for the mechanism of
    # an event and "origin" for the location of an event
    #

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.3, 4.4, 4.5,     
         4.6, 4.7, 4.8]) 

    depths = np.array(
         # depth in meters
        [25000, 30000, 35000, 40000,                    
         45000, 50000, 55000, 60000])         

    sources = DoubleCoupleGridRegular(
        npts_per_axis=25,
        magnitude=magnitudes)

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_FullMomentTensor="""
    #
    # Following obspy, we use the variable name "source" for the mechanism of
    # an event and "origin" for the location of an event
    #

    sources = FullMomentTensorGridRandom(
        npts=1000000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_TestDoubleCoupleMagnitudeDepth="""
    #
    # Next we specify the source parameter grid
    #

    depths = np.array(
         # depth in meters
        [34000])

    sources = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitude=[4.4, 4.5, 4.6])

    wavelet = Trapezoid(
        magnitude=4.5)

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

    sources = [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (on-diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (on-diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (on-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 1., 0., 0.]), # source 4 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 1., 0.]), # source 5 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 0., 1.]), # source 6 (off-diagonal)
       ]

    Mw = 4.5
    M0 = 10.**(1.5*Mw + 9.1) # units: N-m
    for mt in sources:
        mt *= np.sqrt(2)*M0

    wavelet = Trapezoid(
        magnitude=Mw)

"""


Main_GridSearch_DoubleCouple="""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\\n')
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origin = data.get_origins()[0]

        print('Processing data...\\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Green''s functions...\\n')
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\\n')
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
        print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, sources)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, sources)

    if comm.rank==0:
        best_misfit = (results_bw + results_sw).min()
        best_source = sources.get((results_bw + results_sw).argmin())

"""


Main_GridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main I/O work starts now
    #

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.Get_size()

    if rank==0:
        print('Reading data...\\n')
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        print('Processing data...\\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        stations = data.get_stations()
        origin = data.get_origins()[0]

        origins = []
        for depth in depths:
            origins += [origin.copy()]
            setattr(origins[-1], 'depth_in_m', depth)

        print('Reading Green''s functions...\\n')
        greens = download_greens_tensors(stations, origins, model)

        print('Processing Green''s functions...\\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        origins = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    stations = comm.bcast(stations, root=0)
    origins = comm.bcast(origins, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if rank==0:
        print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, sources)

    if rank==0:
        print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, sources)

    if rank==0:
        results = results_bw + results_sw
        best_misfit = (results).min()

        _i, _j = np.unravel_index(np.argmin(results), results.shape)
        best_source = sources.get(_i)
        best_origin = origins[_j]

"""


Main1_SerialGridSearch_DoubleCouple="""
    #
    # The main I/O work starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac',
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Green''s functions...\\n')
    greens = download_greens_tensors(stations, origin, model)

    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)

"""


Main2_SerialGridSearch_DoubleCouple="""
    #
    # The main computational work starts nows
    #

    print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, sources)

    print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, sources)

    best_misfit = (results_bw + results_sw).min()
    best_source = sources.get((results_bw + results_sw).argmin())

"""


Main_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main I/O work starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac', 
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]

    origins = []
    for depth in depths:
        origins += [origin.copy()]
        setattr(origins[-1], 'depth_in_m', depth)

    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Green''s functions...\\n')
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
        data_bw, greens_bw, misfit_bw, origins, sources, verbose=False)

    print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, sources, verbose=False)


"""


Main_TestGraphics="""

    print('Reading data...\\n')
    data = read(path_data, format='sac',
        event_id=event_name,
        tags=['units:cm', 'type:velocity'])

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Green''s functions...\\n')
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

    plot_data_greens(event_name+'.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, mt, header=False)

    print('Figure 2 of 3\\n')

    plot_data_greens(event_name+'.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, mt, header=False)

    print('Figure 3 of 3\\n')

    plot_beachball('test_graphics3.png', mt)

    print('Finished\\n')
"""



Main_TestMisfit="""
    #
    # The main computational work starts nows
    #

    print('Evaluating body wave misfit...\\n')

    results_0 = misfit_bw(
        data_bw, greens_bw, sources, optimization_level=0)

    results_1 = misfit_bw(
        data_bw, greens_bw, sources, optimization_level=1)

    results_2 = misfit_bw(
        data_bw, greens_bw, sources, optimization_level=2)

    print(results_0.max())
    print(results_1.max())
    print(results_2.max())
    print('')


    print('Evaluating surface wave misfit...\\n')

    results_0 = misfit_sw(
        data_sw, greens_sw, sources, optimization_level=0)

    results_1 = misfit_sw(
        data_sw, greens_sw, sources, optimization_level=1)

    results_2 = misfit_sw(
        data_sw, greens_sw, sources, optimization_level=2)

    print(results_0.max())
    print(results_1.max())
    print(results_2.max())
    print('')

"""



WrapUp_GridSearch_DoubleCouple="""
    #
    # Saving results
    #

    if comm.rank==0:
        print('Savings results...\\n')

        plot_data_greens(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source)

        plot_beachball(event_name+'_beachball.png', best_source)

        #grid.save(event_name+'.h5', {'misfit': results})

        print('Finished\\n')

"""


WrapUp_GridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # Saving results
    #

    if comm.rank==0:
        print('Saving results...\\n')

        plot_data_greens(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, best_origin, best_source)

        misfit_vs_depth(event_name+'_misfit_vs_depth_bw.png',
            data_bw, misfit_bw, origins, sources, results_bw)

        misfit_vs_depth(event_name+'_misfit_vs_depth_sw.png',
            data_sw, misfit_sw, origins, sources, results_sw)

        print('Finished\\n')
"""


WrapUp_SerialGridSearch_DoubleCouple="""
    #
    # Saving results
    #

    print('Saving results...\\n')

    plot_data_greens(event_name+'.png', 
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, best_source)

    plot_beachball(event_name+'_beachball.png', best_source)

    #grid.save(event_name+'.h5', {'misfit': results})

    print('Finished\\n')

"""


WrapUp_TestGridSearch_DoubleCouple="""
    best_misfit = (results_bw + results_sw).min()
    best_source = sources.get((results_bw + results_sw).argmin())

    if run_figures:
        plot_data_greens(event_name+'.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_source)

        plot_beachball(event_name+'_beachball.png', best_source)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-8):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            for _a, _b, _bool in zip(
                a, b, np.isclose(a, b, atol=atol, rtol=rtol)):

                print('%s:  %.e <= %.1e + %.1e * %.1e' %\\
                    ('passed' if _bool else 'failed', abs(_a-_b), atol, rtol, abs(_b)))
            print('')

            return np.all(
                np.isclose(a, b, atol=atol, rtol=rtol))

        if not isclose(
            best_source,
            np.array([
                 -2.65479669144e+15,
                  6.63699172860e+14,
                  1.99109751858e+15,
                  1.76986446096e+15,
                  1.11874525051e+00,
                  1.91593448056e+15,
                 ])
            ):
            raise Exception(
                "Grid search result differs from previous mtuq result")

        print('SUCCESS\\n')
"""


WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    best_misfit = (results_bw + results_sw).min()
    best_source = sources.get((results_bw + results_sw).argmin())

    if run_figures:
        filename = event_name+'_misfit_vs_depth.png'
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
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Green''s functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    depth = int(origin.depth_in_m/1000.)+1
    name = '_'.join([model, str(depth), event_name])


    print('Comparing waveforms...')

    for _i, mt in enumerate(sources):
        print('  %d of %d' % (_i+1, len(sources)))

        cap_bw, cap_sw = get_synthetics_cap(
            data_bw, data_sw, paths[_i], name)

        mtuq_bw, mtuq_sw = get_synthetics_mtuq(
            data_bw, data_sw, greens_bw, greens_sw, mt)

        if run_figures:
            plot_data_synthetics('cap_vs_mtuq_'+str(_i)+'.png',
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

        plot_data_synthetics('cap_vs_mtuq_data.png',
            cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
            stations, origin, trace_labels=False, normalize=False)

    print('\\nSUCCESS\\n')

"""


if __name__=='__main__':
    import os
    from mtuq.util import basepath, replace
    os.chdir(basepath())


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
        file.write(Grid_DoubleCouple)
        file.write(Main_GridSearch_DoubleCouple)
        file.write(WrapUp_GridSearch_DoubleCouple)


    with open('examples/GridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'plot_beachball',
            'misfit_vs_depth',
            ))
        file.write(Docstring_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCoupleMagnitudeDepth)
        file.write(Main_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(WrapUp_GridSearch_DoubleCoupleMagnitudeDepth)


    with open('examples/GridSearch.FullMomentTensor.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'FullMomentTensorGridRandom',
            ))
        file.write(Docstring_GridSearch_FullMomentTensor)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(Main_GridSearch_DoubleCouple)
        file.write(WrapUp_GridSearch_DoubleCouple)


    with open('setup/chinook/examples/GridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'syngine',
            'fk',
            ))
        file.write(Docstring_ChinookGridSearch_DoubleCouple)
        file.write(Paths_AxiSEM)
        file.write(
            replace(
            DataProcessingDefinitions,
            'taup_model=.*,',
            'taup_model=\'ak135\',',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(
            replace(
            Main_GridSearch_DoubleCouple,
            'greens = download_greens_tensors\(stations, origin, model\)',
            'db = open_db(path_greens, format=\'AxiSEM\', model=model)\n        '
           +'greens = db.get_greens_tensors(stations, origin)',
            ))
        file.write(WrapUp_GridSearch_DoubleCouple)


    with open('setup/chinook/examples/GridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'syngine',
            'fk',
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'plot_beachball',
            'misfit_vs_depth',
            ))
        file.write(Docstring_ChinookGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(Paths_AxiSEM)
        file.write(
            replace(
            DataProcessingDefinitions,
            'taup_model=.*,',
            'taup_model=\'ak135\',',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCoupleMagnitudeDepth)
        file.write(
            replace(
            Main_GridSearch_DoubleCoupleMagnitudeDepth,
            'greens = download_greens_tensors\(stations, origins, model\)',
            'db = open_db(path_greens, format=\'AxiSEM\', model=model)\n        '
           +'greens = db.get_greens_tensors(stations, origins, verbose=True)',
            ))
        file.write(WrapUp_GridSearch_DoubleCoupleMagnitudeDepth)


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
        file.write(Grid_DoubleCouple)
        file.write(Main1_SerialGridSearch_DoubleCouple)
        file.write(Main2_SerialGridSearch_DoubleCouple)
        file.write(WrapUp_SerialGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            ))
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
            'Random',
            'Regular',
            'npts=.*,',
            'npts_per_axis=5,',
            ))
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
            'verbose=True',
            'verbose=False',
            ))
        file.write(WrapUp_TestGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt_depth.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'plot_beachball',
            'misfit_vs_depth',
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
        file.write(Grid_TestDoubleCoupleMagnitudeDepth)
        file.write(Main_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth)


    with open('tests/test_misfit.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
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
        file.write(
            replace(
            Grid_DoubleCouple,
            'Random',
            'Regular',
            'npts=.*,',
            'npts_per_axis=5,',
            ))
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
            'syngine',
            'fk',
            'plot_data_greens',
            'plot_data_synthetics',
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
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
             ))
        file.write(Docstring_Gallery)
        file.write(Paths_Syngine)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'npts=.*',
            'npts_per_axis=10,',
            ))
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
            'print.*',
            '',
            ))


