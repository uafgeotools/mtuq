#!/usr/bin/env python


import mtuq.misfit.simple
import numpy as np
import unittest
import obspy.core
from mtuq.dataset import Dataset
from mtuq.util import AttribDict
from mtuq.util.signal import pad
from mtuq.wavelet import Gaussian

 
class test_misfit(unittest.TestCase):
    def test_time_shift1(self):
        """ Checks that time-shift corrections are working properly
        """
        npts = 1001
        starttime = -10.
        endtime = 10.
        delta = (endtime-starttime)/(npts-1)
        t = np.linspace(starttime, endtime, npts)

        header = AttribDict()
        header.npts = npts
        header.starttime = starttime
        header.delta = delta
        header.channel = 'Z'
        header.weight = 1.

        #
        # Generates two Gaussian wavelets, identical up to a time shift
        #
        gaussian = Gaussian(sigma=1., mu=0.).evaluate(t)
        dat = Stream(Trace(data=gaussian, header=header))

        time_shift = 0.5
        gaussian = Gaussian(sigma=1., mu=time_shift).evaluate(t)
        syn = Stream(Trace(data=gaussian, header=header))

        #
        # Checks that the correction matches the original shift
        #
        misfit = mtuq.misfit.Misfit(
            time_shift_max=1.)

        result = misfit(data, greens)

        assert hasattr(syn[0][0], 'time_shift')
        assert syn[0][0].time_shift == -time_shift

        #
        # Checks that the time-shift-corrected misfit is zero
        #
        assert np.isclose(result, 0.)
       

    def test_time_shift2(self):
        """ Tests the time_shift_group feature, which allows time shifts to be
            fixed from component to component
        """
        npts = 1001
        starttime = -10.
        endtime = 10.
        delta = (endtime-starttime)/(npts-1)
        t = np.linspace(starttime, endtime, npts)

        header = AttribDict()
        header.npts = npts
        header.starttime = starttime
        header.delta = delta
        header.weight = 1.

        #
        # Generates 6 Gaussian wavelets, all identical up to a time shift, 
        # representing vertical-, radial-, and transverse-component data and
        # synthetics
        #
        dat = Dataset()
        for id in range(1):
            stream = Stream()
            stream.id = id
            for channel in ['Z','R','T']:
                header.channel = channel
                gaussian = Gaussian(sigma=1., mu=0.).evaluate(t)
                stream += Trace(data=gaussian, header=header)
            dat += stream

        syn = Dataset()
        for id in range(1):
            stream = Stream()
            stream.id = id
            for channel in ['Z','R','T']:
                header.channel = channel
                time_shift = np.random.uniform()
                gaussian = Gaussian(sigma=1., mu=time_shift).evaluate(t)
                stream += Trace(data=gaussian, header=header)
            syn += stream

        #
        # Checks that, when time shifts are allowed to vary from component to
        # component, the misfit is less than when time-shifts are fixed
        #
        misfit1 = cap.Misfit(
            time_shift_max=1.,
            time_shift_groups=['Z','R','T'])

        misfit2 = cap.Misfit(
            time_shift_max=1.,
            time_shift_groups=['ZRT'])

        assert misfit1(dat, syn) <= misfit2(dat, syn)


### utility functions

def Stream(*args, **kwargs):
    """ Overloads obspy Stream by seeting the "id" attribute, which 
        mtuq expects (normally this is done by dataset.reader)
    """
    stream = obspy.core.Stream(*args, **kwargs)
    stream.id = 0
    return stream


def Trace(*args, **kwargs):
    """ Overloads obspy Trace by seeting the "weight" attribute, which 
        cap.misfit expects (normally this is done by cap.process_data)
    """
    trace = obspy.core.Trace(*args, **kwargs)
    trace.weight = 1.
    return trace



class GreensTensor(object):

    def __init__(self, synthetics):
        components = []
        for trace in synthetics:
            components += [trace.stats.channel[-1].upper()]

        self.synthetics = synthetics
        self.components = components


    def get_synthetics(self, dummy):
        return self.synthetics

    def _set_components(self, dummy):
        pass



def MomentTensor():
    return []



def misfit(data, synthetics, time_shift_min=0., time_shift_max=0., 
    time_shift_groups=['ZRT'], norm='hybrid'):
    """ Evaluates misfit on a single stream, rather than an entire dataset
    """
    import mtuq.misfit.simple

    for trace in synthetics:
        pad(trace, (time_shift_min, time_shift_max))

    data = [data]
    greens = [GreensTensor(synthetics)]
    sources = [MomentTensor()]

    return mtuq.misfit.simple.misfit(
        data, 
        greens, 
        sources, 
        norm,
        time_shift_groups,
        time_shift_min, 
        time_shift_max, 
        verbose=0,
        set_attributes=True,
        )



if __name__=='__main__':
    #unittest.main()

    npts = 1001
    starttime = -10.
    endtime = 10.
    delta = (endtime-starttime)/(npts-1)
    t = np.linspace(starttime, endtime, npts)

    header = AttribDict()
    header.npts = npts
    header.starttime = starttime
    header.delta = delta
    header.channel = 'Z'
    header.weight = 1.

    #
    # Generates two Gaussian wavelets, identical up to a time shift
    #
    gaussian = Gaussian(sigma=1., mu=0.).evaluate(t)
    dat = Stream(Trace(data=gaussian, header=header))

    time_shift = 1.
    gaussian = Gaussian(sigma=1., mu=-time_shift).evaluate(t)
    syn = Stream(Trace(data=gaussian, header=header))

    #
    # Checks that the correction matches the original shift
    #
    result = misfit(dat, syn, time_shift_min=-2, time_shift_max=+2)


    assert syn[0].time_shift==time_shift
