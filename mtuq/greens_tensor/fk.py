
import obspy
import numpy as np

import mtuq.greens_tensor.base

from collections import defaultdict
from copy import deepcopy
from math import ceil
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.util.signal import resample
from mtuq.util.moment_tensor.change_basis import change_basis


# Precomputed fk Green's functions represent vertical, radial, and transverse
# velocity time series
COMPONENTS = ['Z','R','T']


# For the vertical and raidal components, there are four associated time series.
# For the travserce component, there are two associated time series. Thus there
# ten independent Green's tensor elements altogether, which is fewer than in 
# the case of a general inhomogeneous medium because fk greens functions 
# represent the impulse response of a layered medium.


# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# GreensTensorFactory normally does all the work
ErrorMessage =("An obspy stream must be provided containting 10 traces, each"
    "representing an indepedent Green's tensor element. The order of traces "
    "must match the scheme used by fk. See fk documentation for details.")


DEG2RAD = np.pi/180.



class GreensTensor(mtuq.greens_tensor.base.GreensTensor):
    """
    Elastic Green's tensor object
    """
    def __init__(self, traces, station, origin):
        assert len(traces)==10, ValueError(ErrorMessage)
        super(GreensTensor, self).__init__(traces, station, origin)
        self.tags += ['velocity']


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        for _i, channel in enumerate(self.meta.channels):
            component = channel[-1].upper()
            if component not in COMPONENTS:
                raise Exception("Channels are expected to end in one of the "
                   "following characters: ZRT")
            self._synthetics[_i].meta.channel = component

            # overwrites previous synthetics
            syn = self._synthetics[_i]
            syn.data[:] = 0.

            # linear combination of Green's functions
            for _j, weight in self._calculate_weights(mt, component):
                syn.data += weight*self[_j].data

        return self._synthetics


    def _calculate_weights(self, mt, component):
       """
       Calculates weights used in linear combination over Green's functions

       See cap/fk documentation for indexing scheme details; here we try to
       follow as closely as possible the cap way of doing things

       See also Lupei Zhu's mt_radiat utility
       """
       if component not in COMPONENTS:
           raise Exception

       if not hasattr(self.meta, 'azimuth'):
           raise Exception

       saz = np.sin(DEG2RAD * self.meta.azimuth)
       caz = np.cos(DEG2RAD * self.meta.azimuth)
       saz2 = 2.*saz*caz
       caz2 = caz**2.-saz**2.

       # instaseis/MTUQ use convention 1 (GCMT)
       # CAP/FK uses convention 2 (Aki&Richards)
       mt = change_basis(mt, 1, 2)

       # what weights are used in the linear combination?
       weights = []
       if component in ['R','Z']:
           weights += [(mt[0] + mt[1] + mt[2])/3.]
           weights += [-0.5*caz2*(mt[0] - mt[1]) - saz2*mt[3]]
           weights += [-caz*mt[4] - saz*mt[5]]
           weights += [(2.*mt[2] - mt[0] - mt[1])/6.]

       elif component in ['T']:
           weights += [-0.5*saz2*(mt[0] - mt[1]) + caz2*mt[3]]
           weights += [-saz*mt[4] + caz*mt[5]]

       # what Green's tensor elements do the weights correspond to?
       if component in ['T']:
           indices = [0, 1]
       elif component in ['R']:
           indices = [2, 3, 4, 5]
       elif component in ['Z']:
           indices = [6, 7, 8, 9]

       return zip(indices, weights)


    def _preallocate_synthetics(self):
        """ 
        Creates obspy streams for use by get_synthetics
        """
        self._synthetics = Stream()
        for channel in self.meta.channels:
            self._synthetics +=\
                Trace(np.zeros(self[0].stats.npts), self.meta)
        self._synthetics.id = self.id



class GreensTensorFactory(mtuq.greens_tensor.base.GreensTensorFactory):
    """ 
    Creates a GreensTensorList by reading precomputed Green's tensors from an
    fk directory tree.  Such trees contain SAC files organized by model, event
    depth, and event distance and are associated with the solver package "fk"
    by Lupei Zhu.

    Generating Green's tensors is a two-step procedure:
        1) greens_tensor_generator = mtuq.greens.fk.GreensTensorFactory(path, model)
        2) greens_tensors = greens_tensor_generator(stations, origin)

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
        for ext in ['8','5',          # t
                    'b','7','4','1',  # r
                    'a','6','3','0']: # z
            trace = obspy.read('%s/%s_%s/%s.grn.%s' %
                (self.path, self.model, dep, dst, ext),
                format='sac')[0]

            # what are the start and end times of the Green's function?
            t1_old = float(origin.time)+float(trace.stats.starttime)
            t2_old = float(origin.time)+float(trace.stats.endtime)

            dt_old = float(trace.stats.delta)

            # resample Green's function
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

            stream += trace

        stream.id = station.id

        traces = [trace for trace in stream]
        return GreensTensor(traces, station, origin)

