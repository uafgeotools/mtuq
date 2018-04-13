
import numpy as np
from obspy.core import Trace, Stream, Stats
from mutq.misfit import cap
from mtuq.util.wavelet import Ricker



def test_time_shift1():
    """ Checks that time-shift corrections are working properly
    """
    npts = 1001
    starttime = -10.
    endtime = 10.
    delta = (endtime-starttime)/(npts-1)
    t = np.linspace(starttime, endtime, npts)

    stats = Stats()
    stats.npts = npts
    ststs.starttime = starttime
    stats.delta = delta

    #
    # Generates two Ricker wavelets, identical up to a time shift
    #
    dat = Dataset()
    stream = Stream()
    stream.id = 0
    ricker = Ricker(sigma=1., mu=0.).evaluate(t)
    stream += Trace(data=ricker, stats=stats)
    dat += stream

    syn = Dataset()
    stream = Stream()
    stream.id = 0
    time_shift = 0.5
    ricker = Ricker(sigma=1., mu=time_shift).evaluate(t)
    stream += Trace(data=ricker, stats=stats)
    syn += stream


    #
    # Checks that the correction matches the original shift
    #
    misfit = cap.misfit(
        time_shift_max=1.)

    result = misfit(dat, syn)

    assert hasattr(syn[0][0], 'time_shift')
    assert syn[0][0].time_shift == time_shift


    #
    # Checks that the time-shift-corrected misfit is zero
    #
    assert isclose(result, 0.)
   

    
def test_time_shift2():
    """ Tests the time_shift_group feature, which allows time shifts to be
        fixed from component to component or to vary
    """
    npts = 1001
    starttime = -10.
    endtime = 10.
    delta = (endtime-starttime)/(npts-1)
    stats = Stats(npts=npts, 
    t = np.linspace(starttime, endtime, npts)

    stats = Stats()
    stats.npts = npts
    ststs.starttime = starttime
    stats.delta = delta


    #
    # Generates 6 Ricker wavelets, all identical up to a time shift, 
    # representing vertical-, radial-, and transverse-component data and
    # synthetics
    #
    dat = Dataset()
    for id in range(1):
        stream = Stream()
        stream.id = id
        for channel in ['Z','R','T']:
            stats = Stats()
            stream += Trace(data=Ricker(sigma=1., mu=0.).evaluate(t))
        dat += stream

    syn = Dataset()
    for id in range(1):
        stream = Stream()
        stream.id = id
        for channel in ['Z','R','T']:
            time_shift = np.random.uniform()
            stats = Stats()
            stream += Trace(data=Ricker(sigma=1., mu=time_shift).evaluate(t))
        syn += stream


    #
    # Checks that, when time shifts are allowed to vary from component to
    # component, the misfit is less than when time-shifts are fixed
    #
    misfit1 = cap.misfit(
        time_shift_max=1.,
        time_shift_groups=['ZRT'])

    misfit2= cap.misfit(
        time_shift_max=1.,
        time_shift_groups=['Z','R','T'])

    assert misfit1(dat, syn) <= misfit2(dat, syn)


