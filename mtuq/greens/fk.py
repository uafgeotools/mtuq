
import obspy
import numpy as np
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.greens.base import GreensTensorBase, GreensTensorGeneratorBase,\
    GreensTensorList
from mtuq.util.signal import resample


# fk Green's functions are already rotatated into vertical, radial, and
# transverse components:
COMPONENTS = ['z','r','t']

# for each component, there are four associated time series
N = 4

# Thus there are N*len(COMPONENTS) = 4*3 = 12 independent Green's tensor 
# elements altogether. This is fewer than the number required for a general
# medium, since fk Green's functions represent the impulse response of a
# layered medium


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
    def __init__(self, data, station, origin, mpi=None):
        for key in COMPONENTS:
            assert key in data
            assert len(data[key])==N

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
        dep = str(int(origin.depth/1000.))
        dst = str(int(station.distance))
        # to a vertical strike-slip mechanism (VSS), 6-8 to a horizontal 
        # strike-slip mechanism (HSS), and a-c to an explosive source (EXP); 
        # see "fk" documentation for more details
        keys = [('0','z'),('1','r'),('2','t'),
                ('3','z'),('4','r'),('5','t'),
                ('6','z'),('7','r'),('8','t'),
                ('a','z'),('b','r'),('c','t')]

        # Green's functions will be read into a dictionary
        impulse_response = dict((('z',[]),('r',[]),('t',[])))

        for ext, component in keys:
            # read Green's function
            trace = obspy.read('%s/%s_%s/%s.grn.%s' %
                (self.path, self.model, dep, dst, ext),
                format='sac')[0]

            # start and end times of Green's function
            t1_old = float(origin.time)+float(trace.stats.starttime)
            t2_old = float(origin.time)+float(trace.stats.endtime)
            dt_old = float(trace.stats.delta)

            # start and end times of data
            t1_new = float(station.starttime)
            t2_new = float(station.endtime)
            dt_new = float(station.delta)

            # resample Green's function
            old = trace.data
            new = resample(old, t1_old, t2_old, dt_old, t1_new, t2_new, dt_new)

            impulse_response[component] += [new]

        return GreensTensor(impulse_response, station, origin)


# chinook debugging
if __name__=='__main__':
    from mtuq.io.sac import read, get_origin, get_stations
    data = read('/u1/uaf/rmodrak/packages/capuaf/20090407201255351')
    origin = get_origin(data)
    channels = get_stations(data)
    path = '/center1/ERTHQUAK/rmodrak/data/wf/FK_SYNTHETICS/scak'
    model = 'scak'
    greens_tensor_factory = Factory(path, model)
    greens_tensor_list = greens_tensor_factory(stations, origin)


