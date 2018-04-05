
import obspy
import numpy as np

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.greens_tensor.base import GreensTensorBase, GeneratorBase, GreensTensorList
from mtuq.util.signal import resample


# Green's functions are already rotatated into vertical, radial, and
# transverse components
COMPONENTS = ['z','r','t']

# For each component, there are four associated time series
N = 4

# Thus there are N*len(COMPONENTS) = 4*3 = 12 independent Green's tensor 
# elements altogether.  Because fk Green's functions represent the impulse 
# response of a layered medium, there are fewer indepedent elements than
# in the general case

# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# Generator normally does all the work
ErrorMessage =("An obspy stream must be provided containting 12 traces, each"
    "representing an indepedent Green's tensor element. The order of traces "
    "must match the scheme used by fk. See fk documentation for details.")


class GreensTensor(GreensTensorBase):
    """
    Elastic Green's tensor object
    """
    def __init__(self, stream, station, origin):
        assert isinstance(stream, obspy.Stream), ValueError(ErrorMessage)
        assert len(stream)==N*len(COMPONENTS), ValueError(ErrorMessage)
        super(GreensTensor, self).__init__(stream, station, origin)


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        for _i, channel in enumerate(self.station.channels):
            component = channel[-1].lower()
            if component not in COMPONENTS:
                raise Exception("Channels are expected to end in Z/R/T")

            G = self.greens_tensor
            s = self._synthetics[_i].data

            # overwrite previous synthetics
            s[:] = 0.

            # linear combination of Green's functions
            for _j, weight in self._calculate_weights(mt, component):
                s += weight*G[_j].data

        return self._synthetics


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

       # what weights are used in the linear combination?
       weights = []
       if component in ['r','z']:
           weights += [(2.*mt[2] - mt[0] - mt[1])/6.]
           weights += [-caz*mt[3] - saz*mt[4]]
           weights += [-0.5*caz2*(mt[0] - mt[1]) - saz2*mt[3]]
           weights += [(mt[0] - mt[1] + mt[2])/3.]
       elif component in ['t']:
           weights += [0.]
           weights += [-0.5*saz2*(mt[0] - mt[1]) + caz2*mt[3]]
           weights += [-saz*mt[4] + caz*mt[5]]
           weights += [0.]

       # what Green's tensor elements do the weights correspond to?
       if component in ['z']:
           indices = [0, 1, 2, 3]
       elif component in ['r']:
           indices = [4, 5, 6, 7]
       elif component in ['t']:
           indices = [8, 9, 10, 11]

       return zip(indices, weights)


    def _preallocate_synthetics(self):
        """ 
        Creates obspy streams for use by get_synthetics
        """
        self._synthetics = Stream()
        for channel in self.station.channels:
            self._synthetics +=\
                Trace(np.zeros(self.greens_tensor[0].stats.npts), self.station)
        self._synthetics.id = self.greens_tensor.id



class Generator(GeneratorBase):
    """ 
    Creates a GreensTensorList by reading precomputed Green's tensors from an
    fk directory tree.  Such trees contain SAC files organized by model, event
    depth, and event distance and are associated with the solver package "fk"
    by Lupei Zhu.

    Generating Green's tensors is a two-step procedure:
        1) greens_tensor_generator = mtuq.greens.fk.Generator(path, model)
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

        dep = str(int(origin.depth/1000.))
        dst = str(int(station.distance))

        # see fk documentation for indexing scheme details 
        for ext in ['0','3','6','a',  # z
                    '1','4','7','b',  # r
                    '2','5','8','c']: # t
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

        return GreensTensor(stream, station, origin)

