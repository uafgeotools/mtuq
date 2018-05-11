
GridSearchImports="""
import os
import sys
import numpy as np
import mtuq.dataset.sac
import mtuq.greens_tensor.fk

from os.path import basename, join
from mtuq.grid_search import DCGridRandom, DCGridRegular
from mtuq.grid_search import grid_search_mpi, grid_search_serial
from mtuq.misfit.cap import misfit
from mtuq.process_data.cap import process_data
from mtuq.util.cap_util import trapezoid_rise_time, Trapezoid
from mtuq.util.plot import cap_plot
from mtuq.util.util import AttribDict, root


"""


DocstringDC3Serial="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors, keeping magnitude and depth fixed
    #
    # USAGE
    #   python GridSearchDC3Serial.py
    #
    # A typical runtime is about 60 minutes. For faster results, try 
    # GridSearchDC3Serial.py, which runs the same inversion except in
    # parallel
    #


"""


DocstringDC3="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors, keeping magnitude and depth fixed
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearchDC3.py


"""


DocstringDC5="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 1.25 million randomly chosen double-couple 
    # moment tensors, varying magnitude and depth
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearchDC5.py


"""



DocstringMT5="""
if __name__=='__main__':
    #
    # Full moment tensor inversion example
    # 
    # Carries out a grid search over 5 million randomly chosen moment tensors, 
    # keeping magnitude and depth fixed
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearchDC5.py


"""


CAPBenchmark1="""
if __name__=='__main__':
    # to benchmark against CAPUAF:
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.25/0.6667/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.3 -I1 -R0/0/0/0/180/180/0.5/0.5/0/0 20090407201255351


"""



CAPBenchmark64000="""
if __name__=='__main__':
    # to benchmark against CAPUAF:
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.25/0.6667/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.3 -I1 -R0/0/0/0/180/180/0.5/0.5/0/0 20090407201255351

"""



DefinitionsPaths="""
    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake. For now, these paths exist only in my personal 
    # environment.  Eventually we need to include sample data in the 
    # repository or make it available for download
    #
    paths = AttribDict({
        'data':    join(root(), 'tests/data/20090407201255351'),
        'weights': join(root(), 'tests/data/20090407201255351/weight_test.dat'),
        'greens':  join(os.getenv('CENTER1'), 'data/wf/FK_SYNTHETICS/scak'),
        })

    event_name = '20090407201255351'



"""


DefinitionsDataProcessing="""
    #
    # Here we specify all the data processing and misfit settings used in the
    # inversion.  For this example, body- and surface-waves are processed
    # separately, and misfit is a sum of indepdendent body- and surface-wave
    # contributions. (For a more flexible way of specifying parameters based on
    # command-line argument passing rather than scripting, see
    # mtuq/scripts/cap_inversion.py)
    #

    process_bw = process_data(
        filter_type='Bandpass',
        freq_min= 0.25,
        freq_max= 0.667,
        pick_type='from_fk_database',
        fk_database=paths.greens,
        window_type='cap_bw',
        window_length=15.,
        padding_length=2.,
        weight_type='cap_bw',
        weight_file=paths.weights,
        )

    process_sw = process_data(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_fk_database',
        fk_database=paths.greens,
        window_type='cap_sw',
        window_length=150.,
        padding_length=10.,
        weight_type='cap_sw',
        weight_file=paths.weights,
        )

    process_data = {
       'body_waves': process_bw,
       'surface_waves': process_sw,
       }


"""


DefinitionsMisfit="""
    misfit_bw = misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )

    misfit = {
        'body_waves': misfit_bw,
        'surface_waves': misfit_sw,
        }


"""


GridSearchSerial="""
    #
    # The main work of the grid search starts now
    #

    print 'Reading data...\n'
    data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
    remove_unused_stations(data, paths.weights)
    data.sort_by_distance()

    stations  = []
    for stream in data:
        stations += [stream.station]
    origin = data.get_origin()


    print 'Processing data...\n'
    processed_data = {}
    for key in ['body_waves', 'surface_waves']:
        processed_data[key] = data.map(process_data[key])
    data = processed_data


    print 'Reading Greens functions...\n'
    generator = mtuq.greens_tensor.fk.Generator(paths.greens)
    greens = generator(stations, origin)


    print 'Processing Greens functions...\n'
    greens.convolve(wavelet)
    processed_greens = {}
    for key in ['body_waves', 'surface_waves']:
        processed_greens[key] = greens.map(process_data[key])
    greens = processed_greens


    print 'Carrying out grid search...\n'
    results = grid_search_serial(data, greens, misfit, grid)


    print 'Saving results...\n'
    grid.save(event_name+'.h5', {'misfit': results})
    best_mt = grid.get(results.argmin())


    print 'Plotting waveforms...\n'
    synthetics = {}
    for key in ['body_waves', 'surface_waves']:
        synthetics[key] = greens[key].get_synthetics(best_mt)
    cap_plot(event_name+'.png', data, synthetics, misfit)


"""


GridSearchMPI="""
    #
    # The main work of the grid search starts now
    #
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    if comm.rank==0:
        print 'Reading data...\n'
        data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
        remove_unused_stations(data, paths.weights)
        data.sort_by_distance()

        stations  = []
        for stream in data:
            stations += [stream.station]
        origin = data.get_origin()

        print 'Processing data...\n'
        processed_data = {}
        for key in ['body_waves', 'surface_waves']:
            processed_data[key] = data.map(process_data[key])
        data = processed_data

        print 'Reading Greens functions...\n'
        generator = mtuq.greens_tensor.fk.Generator(paths.greens)
        greens = generator(stations, origin)

        print 'Processing Greens functions...\n'
        greens.convolve(wavelet)
        processed_greens = {}
        for key in ['body_waves', 'surface_waves']:
            processed_greens[key] = greens.map(process_data[key])
        greens = processed_greens

    else:
        data = None
        greens = None

    data = comm.bcast(data, root=0)
    greens = comm.bcast(greens, root=0)


    if comm.rank==0:
        print 'Carrying out grid search...\n'
    results = grid_search_mpi(data, greens, misfit, grid)
    results = comm.gather(results, root=0)


    if comm.rank==0:
        print 'Saving results...\n'
        results = np.concatenate(results)
        grid.save(event_name+'.h5', {'misfit': results})
        best_mt = grid.get(results.argmin())


    if comm.rank==0:
        print 'Plotting waveforms...\n'
        synthetics = {}
        for key in ['body_waves', 'surface_waves']:
            synthetics[key] = greens[key].get_synthetics(best_mt)
        cap_plot(event_name+'.png', data, synthetics, misfit)


"""



GridSearchMPIOrigin="""
    #
    # The main work of the grid search starts now
    #
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    if comm.rank==0:
        print 'Reading data...\n'
        data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
        remove_unused_stations(data, paths.weights)
        data.sort_by_distance()

        stations  = []
        for stream in data:
            stations += [stream.station]
        origin = data.get_origin()

        print 'Processing data...\n'
        processed_data = {}
        for key in ['body_waves', 'surface_waves']:
            processed_data[key] = data.map(process_data[key])
        data = processed_data
    else:
        data = None

    data = comm.bcast(data, root=0)


   for origin in origins:
        if comm.rank==0:
            print 'Reading Greens functions...\n'
            generator = mtuq.greens_tensor.fk.Generator(paths.greens)
            greens = generator(stations, origin)

            print 'Processing Greens functions...\n'
            greens.convolve(wavelet)
            processed_greens = {}
            for key in ['body_waves', 'surface_waves']:
                processed_greens[key] = greens.map(process_data[key])
            greens = processed_greens

        else:
            greens = None

        greens = comm.bcast(greens, root=0)


        if comm.rank==0:
            print 'Carrying out grid search...\n'
        results = grid_search_mpi(data, greens, misfit, grid)
        results = comm.gather(results, root=0)


        if comm.rank==0:
            print 'Saving results...\n'
            results = np.concatenate(results)
            grid.save(event_name+'.h5', {'misfit': results})


        if comm.rank==0:
            print 'Plotting waveforms...\n'
            synthetics = {}
            for key in ['body_waves', 'surface_waves']:
                synthetics[key] = greens[key].get_synthetics(best_mt)
            cap_plot(event_name+'.png', data, synthetics, misfit)


"""




if __name__=='__main__':
    from mtuq.util.util import root
    os.chdir(root())

    with open('examples/GridSearchDC3Serial.py') as fid:
        write(fid, GridSearchImports)
        write(fid, DocstringDC3Serial)
        write(fid, DefinitionsPaths)
        write(fid, DefinitionsDataProcessing)
        write(fid, DefinitionsMisfit)
        write(fid, GridDC3)
        write(fid, GridSearchSerial)


    with open('examples/GridSearchDC3.py') as fid:
        write(fid, GridSearchImports)
        write(fid, DocstringDC3)
        write(fid, DefinitionsPaths)
        write(fid, DefinitionsDataProcessing)
        write(fid, DefinitionsMisfit)
        write(fid, DefinitionsGridDC3)
        write(fid, GridSearchMPI)


    with open('examples/GridSearchDC5.py') as fid:
        write(fid, GridSearchImports)
        write(fid, DocstringDC5)
        write(fid, DefinitionsPaths)
        write(fid, DefinitionsDataProcessing)
        write(fid, DefinitionsMisfit)
        write(fid, GridDC5)
        write(fid, GridSearchMPIOrigin)


    with open('examples/GridSearchMT5.py') as fid:
        write(fid, GridSearchImports)
        write(fid, DocstringDC5)
        write(fid, DefinitionsPaths)
        write(fid, DefinitionsDataProcessing)
        write(fid, DefinitionsMisfit)
        write(fid, GridMT5)
        write(fid, GridSearchMPIOrigin)


    with open('tests/benchmarks/capuaf_npts_1.py') as fid:
        write(fid, GridSearchImports)
        write(fid, DefinitionsPaths)
        write(fid, DefinitionsDataProcessing)
        write(fid, DefinitionsMisfit)
        write(fid, GridCAP1)
        write(fid, GridSearchMPI)
        write(fid, PostprocessCAP)


    with open('tests/benchmarks/capuaf_npts_64000.py') as fid:
        write(fid, GridSearchImports)
        write(fid, DefinitionsPaths)
        write(fid, DefinitionsDataProcessing)
        write(fid, DefinitionsMisfit)
        write(fid, GridCAP64000)
        write(fid, GridSearchMPI)
        write(fid, PostprocessCAP)


