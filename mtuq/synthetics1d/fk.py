
# Preloads Green's functions in the same manner as CAPUAF

from os.path import exists, join

import obspy

from obspy.geodetics import gps2dist_azimuth


def distance(station, origin):
    if hasattr(station, 'latitude') and\
        hasattr(station, 'longitude'):
        dist, azim, bazim = gps2dist_azimuth(
            station.latitude,
            station.longitude,
            origin.latitude,
            origin.longitude)
        return dist/1000.

    else:
        raise Exception


class fk_factory(object):
    def __init__(self, path):
        self.path = path

    def _greens_function(self, origin, station, ij):
        """ Reads Green's function from FK database
        """
        dist_str = str(int(distance(station, origin)))
        depth_str = str(int(origin.depth/1000.))

        # find filename corresponding to precomputed Greens function
        filename = 'scak_%s/%s.grn.%d' % (depth_str, dist_str, ij)
        filename = join(self.path, filename)

        try:
            return obspy.read(filename, format='sac')
        except:
            raise Exception('Error reading file: %s' % filename)


    def __call__(self, origin, stations):
        G = []
        for _i, station in enumerate(stations):
            _tmp = []
            for _j in range(9):
                _tmp += [self._greens(origin, station, _j)]
            G += _tmp
        return G



# debugging
if __name__=='__main__':
    from mtuq.io.sac import read, get_origin, get_stations
    data = read('/u1/uaf/rmodrak/packages/capuaf/20090407201255351')
    origin = get_origin(data)
    stations = get_stations(data)
    generator = fk_factory('/center1/ERTHQUAK/rmodrak/data/wf/FK_SYNTHETICS/scak')
    greens_functions = generator(origin, stations)


