
import os.path
import obspy
import numpy as np
import pickle

from obspy.core import Stream
from obspy.geodetics.base import gps2dist_azimuth
from mtuq.greens_tensor.SPECFEM3D import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample

from mtuq.util_SPECFEM3D import SAMPLING_RATE, DT, SGT_DATABASE_folder, MODEL3D_folder, INFO_GRID_file
from mtuq.util_SPECFEM3D.DSyn import DSyn, RTP_to_DENZ
from mtuq.util_SPECFEM3D.DSGTMgr import DSGTMgr


MT_ELEMENTS = [
    'Mrr',
    'Mtt',
    'Mpp',
    'Mrt',
    'Mrp',
    'Mtp'
]

class Client(ClientBase):
    """ SPECFEM3D Green's tensor client

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

        if not model:
            model = path_or_url
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force

        # database directory or path.
        self.sgt_database_folder = SGT_DATABASE_folder
        self.model3D_folder = MODEL3D_folder
        self.info_grid_file = INFO_GRID_file

        self.sgtMgr = DSGTMgr(self.sgt_database_folder, self.model3D_folder, self.info_grid_file)


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Reads Green's tensors

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)

        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        if station is None:
            raise Exception("Missing station input argument")

        if origin is None:
            raise Exception("Missing station input argument")

        if self.include_mt:
            # Check if the Green's Function (GF) exists,
            # Read from the PKL file storing the GF or generate from SGT.
            prefix = station.id
            GF_file_path = os.path.join(str(self.path), str(prefix)+".PKL")

            b_exist = False
            b_generate = True
            try:
                b_exist = os.path.exists(GF_file_path)
            except:
                b_exist = False

            if b_exist:
                try:
                    with open(GF_file_path, 'rb') as f:
                        stream = pickle.load(f)
                        b_generate = False
                except:
                    b_exist = False
                    b_generate = True
                    try:
                        os.remove(GF_file_path)
                    except:
                        pass

            if b_generate:
                # Generate Green's function from SGT.
                (distance, azimuth, back_azimuth) = gps2dist_azimuth(station.latitude, station.longitude,
                                                                     origin.latitude, origin.longitude)
                sgt = self.sgtMgr.get_sgt(station, origin)
                stream = self.SGT2GF(sgt, back_azimuth)
                stream.id = station.id
                try:
                    with open(GF_file_path, 'wb') as f:
                        pickle.dump(stream, f)
                except:
                    print("! Unable to dump Green's function at {}.".format(GF_file_path))

        if self.include_force:
            raise NotImplementedError


        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # what are the start and end times of the Green's function?
        t1_old = float(origin.time) + float(stream[0].stats.starttime)
        t2_old = float(origin.time) + float(stream[0].stats.endtime)
        dt_old = float(stream[0].stats.delta)

        for trace in stream:
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old,
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            trace.stats.npts = len(data_new)

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'SPECFEM3D',
             ]

        return GreensTensor(traces=[trace for trace in stream],
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)


    def SGT2GF(self, sgt, ba):
        '''Generate Green's Function (GF) from Strain Green's Tensor (SGT) database.'''

        MTs_rtp = np.identity(6)
        stream = Stream()

        for i, mt_rtp in enumerate(MTs_rtp):
            mt_enz = RTP_to_DENZ(mt_rtp)
            # Synthetic waveform in ENZ
            _st = DSyn(mt_enz, sgt, MT_ELEMENTS[i])
            # Rotation (ENZ => RTZ)
            _st.rotate(method='NE->RT', back_azimuth=ba)
            for _tr in _st:
                ch = _tr.stats.channel
                _tr.stats.channel = '%s.%s' % (ch[-1], ch[:3])
                _tr.stats._component = ch[-1]
                _tr.stats.sampling_rate = SAMPLING_RATE
                _tr.stats.delta = DT

            stream += _st
        return stream
