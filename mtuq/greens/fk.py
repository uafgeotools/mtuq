
import obspy
import numpy as np
from copy import deepcopy
from os.path import basename, exists, join

from obspy.core import Stream, Trace
from mtuq.greens.base import GreensTensorBase, GreensTensorList
from mtuq.util.geodetics import distance_azimuth
from mtuq.util.signal import resample
from mtuq.util.util import iterable


# FK Green's functions come already rotatated into vertical, radial, and
# transverse components
COMPONENTS = ['z','r','t']

# number of Green's functions per component
N = 4

# because synthetics are based on wave propagation in a layered medium,
# there are 
#    N * len(COMPONENETS) = 4 * 3 = 12
# Green's tensor elements (fewer than the 18 indepedent elements required for
# general medium)



class GreensTensorFactory(object):
    """ 
    Reads precomputed Green's tensors from a SAC directory tree organized by 
    model, event depth, and event distance.  Such directory layouts are
    associated with the software package "fk" by Lupei Zhu. The resulting 
    Green's tensors are stored in an mtuq GreensTensorList

    Reading Green's tensors is a two-step procedure:
        1) greens_tensor_factory = mtuq.greens.fk.Factory(path, model)
        2) greens_tensor_list = greens_tensor_factory(stations, origin) 

    In the first step, one supplies the path to an fk directory and the name of 
    the corresponding Earth model. (By convention, the name of the model 
    should match the final directory in the path, i.e. model = basename(path)).

    In the second step, one supplies a list of stations and event origin.
    A GreensTensor object will be created for each station-event pair.
    """
    def __init__(self, path=None, model=None):
        """
        Creates a function that can subsequently be called to read 
        Green's tensors

        :str path: path to "fk" directory tree
        :str model: name of Earth model
        """
        if not path:
            raise Exception

        if not exists(path):
            raise Exception

        if not model:
            model = basename(path)

        self.path = path
        self.model = model


    def __call__(self, stations, origin):
        """
        Reads Green's tensors corresponding to given stations and origin
        """
        greens_tensor_list = GreensTensorList()

        for station in stations:
                print station.station # DEBUG

                # add distance and azimuth to station metadata
                station.distance, station.azimuth = distance_azimuth(
                    station, origin)

                # add another GreensTensor to list
                greens_tensor_list += self._read_greens_tensor(
                    station, origin)

                print len(greens_tensor_list[0].data['z'][0])

        return greens_tensor_list


    def _read_greens_tensor(self, station, origin):
        """ 
        Reads a Greens tensor from a directory tree organized by model,
        event depth, and event distance

        :input station: station information dictionary
        :type station: obspy.core.trace.Stats
        :input origin: event information dictionary
        :type origin: obspy.core.event.Origin
        """
        dep = str(int(origin.depth/1000.))
        dst = str(int(station.distance))
        t1 = station.starttime
        t2 = station.endtime
        dt = station.delta

        # 0-2 correspond to dip-slip mechanism (DS)
        # 3-5 correspond to vertical strike-slip mechanism (VSS)
        # 6-8 correspond to horizontal strike-slip mechanism (HSS)
        # a-c correspond to an explosive source (EXP)
        # z,r,t stand for vertical,radial,transverse respectively
        # see "fk" documentation for more details
        keys = [('0','z'),('1','r'),('2','t'),
                ('3','z'),('4','r'),('5','t'),
                ('6','z'),('7','r'),('8','t'),
                ('a','z'),('b','r'),('c','t')]

        # Green's functions will be read into a dictionary indexed by component
        data = dict((('z',[]),('r',[]),('t',[])))

        for ext, component in keys:
            filename = '%s/%s_%s/%s.grn.%s' %\
                (self.path, self.model, dep, dst, ext)
            trace = obspy.read(filename, format='sac')[0]
            trace_resampled = resample(trace, t1, t2, dt)
            data[component] += [trace_resampled]

        return GreensTensor(data, station, origin)



class GreensTensor(GreensTensorBase):
    """ Adds machinery for generating synthetics
    """
    def __init__(self, data, station, origin, mpi=None):
        # data must be a dictionary indexed by component (z,r,t)
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

        # preallocate streams for use by get_synthetics
        self._synthetics = []
        for _ in range(nproc):
            self._synthetics += [Stream()]
            for channel in station.channels:
                self._synthetics[-1] += Trace(np.zeros(station.npts), station)


    def _calculate_weights(self, mt, component):
       """
       Calculates weights used in linear combination over Green's functions

       See also Lupei Zhu's mt_radiat utility
       """
       return [0.,0.,0.,0.]
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


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via linear
        combination of Green's functions

        :input mt: moment tensor
        :type mt: array, length 6
        """
        if self.mpi:
            iproc = self.mpi.COMM.rank
        else:
            iproc = 0

        for _i, channel in enumerate(self.station.channels):
            component = channel[-1].lower()
            if component not in COMPONENTS:
                raise Exception(
                    "Channels must follow expected naming convention")

            w = self._calculate_weights(mt, component)
            G = self.data[component]
            s = self._synthetics[iproc][_i].data

            # overwrites previous synthetics
            s[:] = 0.
            for _i in range(N):
                s += w[_i]*G[_i]

        return self._synthetics[iproc]


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's functions
        """
        # NOTE: overwrites self.data
        for _c in ['z','r','t']:
            for _i in range(4):
                self.data[_c][_i] =\
                    function(self.data[_c][_i], *args, **kwargs)
        return self



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


