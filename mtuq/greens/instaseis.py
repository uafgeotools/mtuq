
import obspy
import numpy as np
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.greens.base import GreensTensorBase, GreensTensorGeneratorBase,\
    GreensTensorList
from mtuq.util.geodetics import distance_azimuth
from mtuq.util.signal import resample


class GreensTensor(GreensTensorBase):
    """
    Elastic Green's tensor object.  Similar to an obpy Trace, except rather 
    than a single time series, holds multiple time series corresponding to
    the independent elements of an elastic Green's tensor
    """
    def __init__(self, data, station, origin, mpi=None):
        self.data = data
        self.station = station
        self.origin = origin

        self.mpi = mpi
        if self.mpi:
            nproc = self.mpi.COMM.size
        else:
            nproc = 1

        # preallocate streams used by get_synthetics
        self._synthetics = []
        for _ in range(nproc):
            self._synthetics += [Stream()]
            for channel in station.channels:
                self._synthetics[-1] += Trace(np.zeros(station.npts), station)


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if self.mpi:
            iproc = self.mpi.COMM.rank
        else:
            iproc = 0

        for _i, channel in enumerate(self.station.channels):
            component = channel[-1].lower()
            if component not in COMPONENTS:
                raise Exception("Channels are expected to end in Z,R,T")

            w = self._calculate_weights(mt, component)
            G = self.data[component]
            s = self._synthetics[iproc][_i].data

            # overwrites previous synthetics
            s[:] = 0.
            for _j in range(N):
                s += w[_j]*G[_j]

        return self._synthetics[iproc]


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's functions
        """
        # overwrites original data with processed data
        for _c in ['z','r','t']:
            for _i in range(4):
                self.data[_c][_i] =\
                    function(self.data[_c][_i], *args, **kwargs)
        return self


    def _calculate_weights(self, mt, component):
       """
       Calculates weights used in linear combination over Green's functions

       See also Lupei Zhu's mt_radiat utility
       """
       if component not in COMPONENTS:
           raise Exception

       if not hasattr(self.station, 'azimuth'):
           raise Exception

       saz = np.sin(self.station.azimuth)
       caz = np.cos(self.station.azimuth)
       saz2 = 2.*saz*caz
       caz2 = caz**2.-saz**2.

       weights = []
       if component in ['r','z']:
           weights += [(2.*mt[2] - mt[0] - mt[1])/6.]
           weights += [-caz*mt[3] - saz*mt[4]]
           weights += [-0.5*caz2*(mt[0] - mt[1]) - saz2*mt[3]]
           weights += [(mt[0] - mt[1] + mt[2])/3.]
           return weights
       elif component in ['t']:
           weights += [0.]
           weights += [-0.5*saz2*(mt[0] - mt[1]) + caz2*mt[3]]
           weights += [-saz*mt[4] + caz*mt[5]]
           weights += [0.]
           return weights



class GreensTensorGenerator(GreensTensorGeneratorBase):
    """ 
    Creates a GreensTensorList by reading precomputed Green's tensors from an
    fk directory tree.  Such trees contain SAC files organized by model, event
    depth, and event distance and are associated with the solver package "fk"
    by Lupei Zhu.

    Generating Green's tensors is a two-step procedure:
    1) greens_tensor_generator = mtuq.greens.fk.GreensTensorGenerator(path, model)
    2) greens_tensor_list = greens_tensor_generator(stations, origin)

    In the first step, the user supplies the path to an fk directory tree and 
    the name of the  layered Earth model that was used to generate Green's
    tensors contained in the tree.

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.
    """
    def __init__(self, path, buffer_size_in_mb=100.):
        if not exists(path):
            raise Exception

        self.db = ForwardInstaseisDB(
            path, buffer_size_in_mb)



    def get_greens_tensor(self, station, origin):
        """ 
        Reads a Greens tensor from a directory tree organized by model,
        event depth, and event distance
        """
        for i in range(N):
            data += [self.db.get_seismogram(
                _receiver(self.stats),
                _source(mt, self.origin),
                _component(self.stats))]

        return GreensTensor(data, station, origin)


