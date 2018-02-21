
import obspy
import numpy as np

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.greens.base import GreensTensorBase, GreensTensorGeneratorBase,\
    GreensTensorList
from mtuq.util.signal import resample
from mtuq.util.util import is_mpi_env


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
# GreensTensorGenerator normally does all the work
ErrorMessage =("Green's functions must be given as a dictionary indexed by"
    "component (z,r,t)")


class GreensTensor(GreensTensorBase):
    """
    Elastic Green's tensor object.  Similar to an obpy Trace, except rather 
    than a single time series, holds multiple time series corresponding to
    the independent elements of an elastic Green's tensor

    To create a GreensTensor, a dictionary "data" must be supplied. This
    dictionary must be indexed by component (z,r,t), which is a natural way of
    organizing fk Green's functions. In practice, users rarely have to worry 
    about these details because GreensTensorGenerator normally does all the 
    work of creating GreensTensors
    """
    def __init__(self, data, station, origin):
        for key in COMPONENTS:
            # check that input data matches expected format
            assert key in data, ErrorMessage
            assert len(data[key])==N, ErrorMessage

        self.data = data
        self.station = station
        self.origin = origin
        self._synthetics = []


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if is_mpi_env():
            from mpi4py import MPI
            iproc = MPI.comm.rank
        else:
            iproc = 0
        if not self._synthetics:
            self._preallocate_synthetics()

        for _i, channel in enumerate(self.station.channels):
            component = channel[-1].lower()
            if component not in COMPONENTS:
                raise Exception("Channels are expected to end in Z, R, or T")

            w = self._calculate_weights(mt, component)
            G = self.data[component]
            s = self._synthetics[iproc][_i].data

            # overwrites previous synthetics
            s[:] = 0.
            for _j in range(N):
                s += w[_j]*G[_j].data

        return self._synthetics[iproc]


    def apply(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's functions
        """
        processed = defaultdict(Stream)
        for component in ['z','r','t']:
            processed[component] = function(self.data[component], *args, **kwargs)

        # updates metadata in case time sampling changed
        station, origin = self._update_time_sampling(processed)

        return GreensTensor(processed, station, origin)


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


    def _preallocate_synthetics(self):
        """ 
        Creates obspy streams for use by get_synthetics
        """
        if is_mpi_env():
            # every MPI process needs its own stream
            from mpi4py import MPI
            nproc = MPI.comm.size
        else:
            nproc = 1

        self._synthetics = []
        for _ in range(nproc):
            self._synthetics += [Stream()]
            for channel in self.station.channels:
                self._synthetics[-1] +=\
                    Trace(np.zeros(self.station.npts), self.station)


    def _update_time_sampling(self, processed_data):
        """ 
        Checks if time sampling has been affected by data processing and makes
        any required metadata updates
        """
        station, origin = deepcopy(self.station), deepcopy(self.origin)
        stats = processed_data['z'][0].stats

        station.npts = stats.npts
        station.starttime = stats.starttime
        station.delta = stats.delta

        return station, origin



class GreensTensorGenerator(GreensTensorGeneratorBase):
    """ 
    Creates a GreensTensorList by reading precomputed Green's tensors from an
    fk directory tree.  Such trees contain SAC files organized by model, event
    depth, and event distance and are associated with the solver package "fk"
    by Lupei Zhu.

    Generating Green's tensors is a two-step procedure:
    1) greens_tensor_generator = mtuq.greens.fk.GreensTensorGenerator(path, model)
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

        self.path = path
        self.model = model


    def get_greens_tensor(self, station, origin):
        """ 
        Reads a Greens tensor from a directory tree organized by model, event
        depth, and event distance
        """
        # Green's tensor elements are tracess; will be stored into a dictionary
        # based on component
        greens_tensor = defaultdict(lambda: Stream())

        # event information
        dep = str(int(origin.depth/1000.))
        dst = str(int(station.distance))

        # start and end times of data
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # 0-2 correspond to a vertical strike-slip mechanism (VSS), 6-8 to a
        # horizontal strike-slip mechanism (HSS), and a-c to an explosive 
        # source (EXP); see fk documentation for details
        keys = [('0','z'),('1','r'),('2','t'),
                ('3','z'),('4','r'),('5','t'),
                ('6','z'),('7','r'),('8','t'),
                ('a','z'),('b','r'),('c','t')]

        for ext, component in keys:
            # read Green's function
            trace = obspy.read('%s/%s_%s/%s.grn.%s' %
                (self.path, self.model, dep, dst, ext),
                format='sac')[0]

            # start and end times of Green's function
            t1_old = float(origin.time)+float(trace.stats.starttime)
            t2_old = float(origin.time)+float(trace.stats.endtime)
            dt_old = float(trace.stats.delta)

            # resample Green's function
            old = trace.data
            new = resample(old, t1_old, t2_old, dt_old, t1_new, t2_new, dt_new)
            trace.data = new
            trace.stats.arrival_P_fk = t1_old
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            station.arrival_P_fk = t1_old

            greens_tensor[component] += trace

        return GreensTensor(greens_tensor, station, origin)

