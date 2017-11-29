
# NOT IMPLEMENTED
# - index_map
# - cut
# - rotate
# - stats


from os.path import exists, join

import obspy

from mtuq.greens import GreensTensor
from mtuq.util.geodetics import distance


# maps from FK tensor indices to MTUQ tensor indices
index_map =\
    [0, 1, 2, 3, 4, 5]


class generator(object):
    """ 
    Reads Green's tensor from FK database into a GreensTensor object.
    (Works in a similar manner as uafseismo/capuaf.)

    param path: path to FK database
    param model: name of the Earth model
    """
    def __init__(self, path, model=None, keep_stations=):
        if not exists(path):
            raise Exception

        if not model:
            model = basename(path)

        self.path = path
        self.model = model


    def read(self, station, origin):
        """ 
        Reads Green's tensor
        """
        # Greens functions are stored as as SAC files in a directory tree 
        # organized by model, event depth, and event distance
        depth = str(int(origin.depth/1000.))
        distance = str(int(distance(station, origin)))

        # read Green's tensors
        tensor = []
        for _i in index_map:
            filename = '%s_%s/%s.grn.%d' % (self.model, depth, distance, _i)
            fullname = join(self.path, filename)
            try:
                tensor += [obspy.read(fullname, format='sac')[0].data]
            except:
                raise Exception('Error reading file: %s' % filename)

        return data


    def __call__(self, stations, origin, rotate=True):
        """ 
        Reads precomputed Green's tensors corresponding to a given
        channels list and origin
        """
        tensors = GreensTensorList()

        for stats in stations:
            station = '.'.join([stats.network, stats.station])

            if station != stations[-1]:
                tensor = self.read(station, origin)

            # cut and resample time series
            for time_series in tensor:
                tensor = cut(tensor, stats.sample_rate, stats.starttime, stats.endtime)

            # rotate tensor
            rotated = rotate(tensor, stats)

            tensors += GreensTensor(rotated, stats)
            stations += [station]



def iterable(*args):
    return args
      


# debugging
if __name__=='__main__':
    from mtuq.io.sac import read, get_origin, get_stations
    data = read('/u1/uaf/rmodrak/packages/capuaf/20090407201255351')
    origin = get_origin(data)
    channels = get_stations(data)
    path = '/center1/ERTHQUAK/rmodrak/data/wf/FK_SYNTHETICS/scak'
    model = 'scak'
    generator = generator(path, model)
    greens = generator(channels, origin)

