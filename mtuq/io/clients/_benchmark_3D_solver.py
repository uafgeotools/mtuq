
import obspy
import numpy as np

from glob import glob
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth
from os.path import basename
from mtuq.greens_tensor._benchmark_3D_solver import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util import warn
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
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force

        self._tree1 = self._parse1()
        self._tree2 = dict()

        self._prefix1 = 'depth_in_m'
        self._prefix2 = 'offset_in_m'


    def _parse1(self):
        wildcard = self.path+'/*'
        keys = [basename(item).split('-')[1] for item in sorted(glob(wildcard))]
        vals = [float(key) for key in keys]
        return dict(zip(keys, vals))

    def _parse2(self, depth):
        wildcard = self.path+'/'+self._prefix1+'-'+depth+'/*'
        keys = [basename(item).split('-')[1] for item in sorted(glob(wildcard))]
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
        vals = list(tree2[depth_key].values())
        idxmin = np.argmin(np.abs(np.array(vals) - offset))
        offset_key = keys[idxmin]
        offset_val = vals[idxmin]

        return '%s/%s-%s/%s-%s/' %\
            (self.path, self._prefix1, depth_key, self._prefix2, offset_key)


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

        try:
            distance_in_m = np.linalg.norm(np.array([
                station.offset_x_in_m - origin.offset_x_in_m,
                station.offset_y_in_m - origin.offset_y_in_m]))

            warn("Using x,y coordinate system")

        except:
            distance_in_m, _, _ = gps2dist_azimuth(
                origin.latitude,
                origin.longitude,
                station.latitude,
                station.longitude)

            warn("Using lat,lon coordinate system")


        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # find path of nearest Green's tensor
        path = self.find_nearest(distance_in_m, origin.depth_in_m)

        if self.include_mt:
            for _i, ext in enumerate(EXTENSIONS):
                #trace = obspy.read('%s/%s' %  (path, ext),
                #    format='sac')[0]

                channel = ext
                component = ext[0]

                from obspy.core import Stats, Trace
                fromfile = np.loadtxt('%s/%s' %  (path, ext))
                t, data = fromfile[:,0], fromfile[:,1]
                trace = Trace(data, header=Stats({'starttime':t[0], 'npts':len(t), 'delta':t[1]-t[0]}))
                #if component in ['R', 'T', 'E', 'N', '1', '2']:
                #    trace.data *= -1.

                trace.stats.channel = channel
                trace.stats._component = component

                # what are the start and end times of the Green's function?
                t1_old = float(origin.time)+float(trace.stats.starttime)
                t2_old = float(origin.time)+float(trace.stats.endtime)
                dt_old = float(trace.stats.delta)
                data_old = trace.data

                # resample Green's function
                data_new = resample(data_old, t1_old, t2_old, dt_old,
                                              t1_new, t2_new, dt_new)

                # convert to Newtons
                # FIXME: is this the correct scaling?
                #data_new *= 1.e-10

                trace.data = data_new
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

