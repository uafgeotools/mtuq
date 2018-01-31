
import obspy
import mtuq.greens.base

from os.path import basename, exists, join
from mtuq.util.geodetics import distance_azimuth
from mtuq.util.util import iterable


class GreensTensorFactory(object):
    """ 
    Reads precomputed Green's tensors from a SAC directory tree organized by 
    model, event depth, and event distance.  Such directory layouts are
    associated with the software package "fk" by Lupei Zhu. The resulting 
    Green's tensors are stored in an mtuq GreensTensorList, similar to how 
    traces are stored in an obspy Stream.

    Reading Green's tensors is a two-step procedure:
        1) greens_tensor_factory = mtuq.greens.fk.Factory(path, model)
        2) greens_tensor_list = greens_tensor_factory(stations, origins) 

    In the first step, one supplies the path to an fk directory and the name of 
    the corresponding Earth model. (By convention, the name of the model 
    should match the final directory in the path, i.e. model = basename(path)).

    In the second step, one supplies a list of stations and event origins.
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


    def __call__(self, stations, origins):
        """
        Reads Green's tensors corresponding to given stations and origins
        """
        greens_tensor_list = mtuq.greens.base.GreensTensorList()

        for origin in iterable(origins):
            for station in stations:
                    station.distance, station.azimuth = distance_azimuth(
                        station, origin)
                    greens_tensor_list += _read_greens_tensor(
                        self.path, self.model, station, origin)

        return greens_tensor_list


class GreensTensor(mtuq.greens.base.GreensTensor):
    """ Modifies GreensTensor base class, adding machinery for generating 
      synthetics
    """
    def _calculate_weights(self, mt):
       """
       Calculates weights needed for generating synthetics in a linear 
       combination over Green's tensor elements

       See also Lupei Zhu's mt_radiat utility
       """
       if not hasattr(self.stats, 'channel'):
           raise Exception

       if self.stats.channel not in ['r','t','z']:
           raise Exception

       if not hasattr(self.stats, 'azimuth'):
           raise Exception

       saz = np.sin(self.stats.azimuth)
       caz = np.cos(self.stats.azimuth)
       saz2 = 2.*saz*caz
       caz2 = caz**2.-saz**2.

       if self.stats.channel in ['r','R','z','Z']:
           weights += [(2.*mt[2] - mt[0] - mt[1])/6.]
           weights += [-caz*mt[3] - saz*mt[4]]
           weights += [-0.5*caz2*(m[0] - m[1]) - saz2*m[3]]
           weights += [(mt[0] - mt[1] + mt[2])/3.]
           return weights

       if self.stats.channel in ['t','T']:
           weights += [0.]
           weights += [-0.5*saz2*(m[0] - m[1]) + caz2*m[3]]
           weights += [-saz*m[4] + caz*m[5]]
           weights += [0.]
           return weights


    def combine(self, mt):
        """
        Generates synthetic seismogram corresponding to a given moment tensor,
        via linear combination of Green's tensor elements

        :input mt: moment tensor
        :type mt: array, length 6
        """
        nt = gt[0].size
        syn = np.zeros(nt)

        channel = self.stats.channel
        weights = self._calculate_weights(mt)
        greens_functions = self.data[channel]
        for _i in range(4):
            syn += weights[_i]*greens_functions[_i]
        return syn


def _read_greens_tensor(path, model, stats, origin):
    """ 
    Reads a Greens tensor from a directory tree organized by model,
    event depth, and event distance

    :input station: station information dictionary
    :type station: obspy.core.trace.Stats
    :input origin: event information dictionary
    :type origin: obspy.core.event.Origin
    """
    dep = str(int(origin.depth/1000.))
    dst = str(int(stats.distance))

    # in the following list,
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

    # organizes Green's tensor elements by component (z,r,t) and 
    # source type (DS,VSS,HZZ,EXP)
    data = dict((('z',[]),('r',[]),('t',[])))
    for ext, component in keys:
        filename = '%s/%s_%s/%s.grn.%s' % (path, model, dep, dst, ext)
        data[component] += [_read_sac(filename)]

    return GreensTensor(data, stats)


def _read_sac(filename):
    try:
        stream = obspy.read(filename, format='sac')
        return stream[0].data
    except:
        raise Exception('Error reading SAC file: %s' % filename)


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

