
import obspy
import numpy as np

import mtuq.greens_tensor.base
import mtuq.greens_tensor.instaseis


from collections import defaultdict
from copy import deepcopy
from math import ceil
from os.path import basename, exists

from obspy.core import Stream
from mtuq.util.signal import resample
from mtuq.util.moment_tensor.change_basis import change_basis


# fk Green's functions represent vertical, radial, and transverse
# velocity time series (units: 10^-20 cm (dyne-cm)^-1 s^-1) 
COMPONENTS = ['Z', 'R','T']


# fk Green's functions describe the impulse response of a horizontally layered 
# medium. For the vertical and radial components, there are four associated 
# time series. For the tranverse component, there are two associated time 
# series. Thus there ten independent Green's tensor elements altogether, 
# which is fewer than in the case of a general inhomogeneous medium


# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# GreensTensorFactory normally does all the work
ErrorMessage =("A list of 10 traces must be provided, each representing an"
    "indepedent Green's tensor element. The order of traces must match the "
    "scheme used by fk. See fk documentation for details.")


DEG2RAD = np.pi/180.



class GreensTensor(mtuq.greens_tensor.instaseis.GreensTensor):
    """
    Elastic Green's tensor object
    """
    def __init__(self, traces, station, origin):
        #assert len(traces)==10, ValueError(ErrorMessage)
        super(GreensTensor, self).__init__(traces, station, origin)
        self.components = COMPONENTS
        self.tags += ['velocity']


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        # This moment tensor permutation produces a match between mtuq 
        # and cap synthetics.  But what basis conventions does it actually
        # represent?
        Mxx =  mt[1]
        Myy =  mt[2]
        Mzz =  mt[0]
        Mxy =  mt[5]
        Mxz = -mt[3]
        Myz =  mt[4]

        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        if not hasattr(self, '_weighted_tensor'):
            self._precompute_weights()

        for _i, component in enumerate(self.components):
            # which Green's functions correspond to given component?
            if component=='Z':
                _j=0
            elif component=='R':
                _j=1
            elif component=='T':
                _j=2
            G = self._weighted_tensor[_j]

            # we could use np.dot instead, but speedup appears negligible
            s = self._synthetics[_i].data
            s[:] = 0.
            s += Mxx*G[:,0]
            s += Myy*G[:,1]
            s += Mzz*G[:,2]
            s += Mxy*G[:,3]
            s += Mxz*G[:,4]
            s += Myz*G[:,5]

        return self._synthetics



class GreensTensorFactory(mtuq.greens_tensor.base.GreensTensorFactory):
    """ 
    Creates a GreensTensorList by reading precomputed Green's tensors from an
    fk directory tree.  Such trees contain SAC files organized by model, event
    depth, and event distance and are associated with the solver package "fk"
    by Lupei Zhu.

    Generating Green's tensors is a two-step procedure:
        1) factory = mtuq.greens.fk.GreensTensorFactory(path, model)
        2) greens_tensors = factory(stations, origin)

    In the first step, the user supplies the path to an fk directory tree and 
    the name of the  layered Earth model that was used to generate Green's
    tensors contained in the tree.

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.
    """
    def __init__(self, path=None, model=None):
        if not path:
            raise Exception

        if not exists(path):
            raise Exception

        if not model:
            model = basename(path)

        # path to fk directory tree
        self.path = path

        # model from which fk Green's functions were computed
        self.model = model


    def get_greens_tensor(self, station, origin):
        """ 
        Reads a Greens tensor from a directory tree organized by model, event
        depth, and event distance
        """
        stream = Stream()

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        dep = str(int(round(origin.depth/1000.)))
        #dst = str(int(round(station.distance)))
        dst = str(int(ceil(station.distance)))

        # See cap/fk documentation for indexing scheme details;
        # here we try to follow as closely as possible the cap way of
        # doing things
        channels = [
            'TSS', 'TDS',
            'REP', 'RSS', 'RDS', 'RDD',
            'ZEP', 'ZSS', 'ZDS', 'ZDD',
            ]

        extensions = [
            '8','5',           # t
            'b','7','4','1',   # r
            'a','6','3','0',   # z
            ]

        for _i, ext in enumerate(extensions):
            trace = obspy.read('%s/%s_%s/%s.grn.%s' %
                (self.path, self.model, dep, dst, ext),
                format='sac')[0]

            trace.stats.channel = channels[_i]

            # what are the start and end times of the Green's function?
            t1_old = float(origin.time)+float(trace.stats.starttime)
            t2_old = float(origin.time)+float(trace.stats.endtime)

            dt_old = float(trace.stats.delta)

            # resample Green's function
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                t1_new, t2_new, dt_new)
            trace.data = data_new
            # convert from 10^-20 dyne to N^-1
            trace.data *= 1.e-15
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

            stream += trace

        stream.id = station.id

        traces = [trace for trace in stream]
        return GreensTensor(traces, station, origin)

