
import obspy
import numpy as np

from obspy.core import Stream
from mtuq.greens_tensor._benchmark_3D_solver import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample


EXTENSIONS = [
    'Z.Mrr',
    'Z.Mtt',
    'Z.Mpp',
    'Z.Mrt',
    'Z.Mrp',
    'Z.Mtp',
    'R.Mrr',
    'R.Mtt',
    'R.Mpp',
    'R.Mrt',
    'R.Mrp',
    'R.Mtp',
    'T.Mrr',
    'T.Mtt',
    'T.Mpp',
    'T.Mrt',
    'T.Mrp',
    'T.Mtp',
    ]


class Client(ClientBase):
    """ For the special case where SPECFEM3D is used with a 1D model

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.SPECFEM3D_SAC import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

    """

    def __init__(self, path_or_url=None, model=None, 
                 include_mt=True, include_force=False):

        self.path = path_or_url

        self.include_mt = include_mt
        self.include_force = include_force

        self._tree1 = self._parse1()
        self._tree2 = dict()


    def _parse1(self):
        keys = [basename(item) for items in sorted(glob(self.path+'/*'))]
        vals = [float(key) for key in keys]
        return dict(zip(keys, vals))

    def _parse2(self, depth):
        keys = [basename(item) for items in sorted(glob(self.path+'/*/'+depth))]
        vals = [float(key) for key in keys]
        return dict(zip(keys, vals))

    def find_nearest(self, offset, depth):
        tree1 = self._tree1
        tree2 = self._tree2

        # find nearest available depth
        keys = list(tree1.keys())
        vals = list(tree1.values())
        idxmin = np.argmin(np.abs(np.array(vals) - depth))
        depth_key = keys[idxmin]
        depth_val = vals[idxmin]

        if depth_key not in tree2:
            tree2[depth_key] = self._parse2(depth_key)

        # find nearest available offset
        keys = list(tree2[depth_key].keys())
        vals = list(tree2[depth_key].vals())
        idxmin = np.argmin(np.abs(np.array(vals) - offset))
        offset_key = keys[idxmin]
        offset_val = vals[idxmin]

        return '%s/%s/%s/' % (self.path, depth_key, offset_key)


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Reads Green's tensors

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        :param stations: List of ``mtuq.Station`` objects
        :param origins: List of ``mtuq.Origin`` objects
        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        if station is None:
            raise Exception("Missing station input argument")

        if origin is None:
            raise Exception("Missing station input argument")

        traces = []

        distance_in_m, _, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # find path of nearest Green's tensor
        path = self.find_nearest(distance_in_m, origin.depth_in_m)

        if self.include_mt:
            for _i, ext in enumerate(EXTENSIONS):
                trace = obspy.read('%s/%s' %  (path, ext),
                    format='sac')[0]

                trace.stats.channel = CHANNELS[_i]
                trace.stats._component = CHANNELS[_i][0]

                # what are the start and end times of the Green's function?
                t1_old = float(origin.time)+float(trace.stats.starttime)
                t2_old = float(origin.time)+float(trace.stats.endtime)
                dt_old = float(trace.stats.delta)
                data_old = trace.data

                # resample Green's function
                data_new = resample(data_old, t1_old, t2_old, dt_old,
                                              t1_new, t2_new, dt_new)
                trace.data = data_new
                # convert from 10^-20 dyne to N^-1
                trace.data *= 1.e-15
                trace.stats.starttime = t1_new
                trace.stats.delta = dt_new

                traces += [trace]

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'SPECFEM3D',
             ]

        return GreensTensor(traces=[trace for trace in traces],
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)


