
import os.path
import seisgen
from seisclient.seisclient import SeisClient
import obspy
import numpy as np
import pickle

from mtuq.greens_tensor.SPECFEM3D import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample


class Client(ClientBase):
    """ Client obtaining the SPECFEM3D strain Green's tensor from a remote database https://seis.cloud

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.seiscloud import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note ::

        For instructions on creating SPECFEM3D/3D_GLOBE strain Green's tensor
        databases, see `SEISGEN documentation
        <https://github.com/Liang-Ding/seisgen>`_

        For information of SeisCloud and SeisClient, see:
        <https://seis.cloud>`_
        <https://github.com/Liang-Ding/seisclient>`_


    """

    def __init__(self, path_or_url=None, model=None,
                 include_mt=True, include_force=False):

        self.path = path_or_url

        if not model:
            model = path_or_url
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force

        self.b_initial_db = False
        self.b_new_origin = True
        self.origin = 0

        # use the default model name to initialize seisclient.
        self.set_remote_db()

    def set_remote_db(self, model=None):
        """ Set and utilize the remote database. """
        self.seiscloud = SeisClient()
        self.b_initial_db = True


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
            data_type = '3DGF'
            fmt = 'PKL'
            GF_name = '%s.%.6f.%.6f.%.2f.%s.%s' % (prefix,
                                                   np.round(origin.latitude, 6),
                                                   np.round(origin.longitude, 6),
                                                   np.round(origin.depth_in_m, 2),
                                                   data_type,
                                                   fmt)
            GF_file_path = os.path.join(str(self.path), str(GF_name))

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
                # Download (generate) Green's function from SGT.
                if not self.b_initial_db:
                    raise Exception("Remote database is not initialized!")

                if not self.b_new_origin:
                    try:
                        if origin.latitude != self.origin.latitude or origin.longitude != self.origin.longitude:
                            self.b_new_origin = True
                        else:
                            try:
                                if origin.depth_in_m != self.origin.depth_in_m:
                                    self.b_new_origin = True
                            except:
                                if origin.elevation_in_m != self.origin.elevation_in_m:
                                    self.b_new_origin = True
                    except:
                        self.b_new_origin = True

                stream = self.seiscloud.get_greens_3DMT(model=self.model, station=station, origin=origin, save_dir=self.path)

                if self.b_new_origin:
                    self.origin = origin
                    self.b_new_origin = False

                try:
                    # save 3D Greens functions to file.
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
            'solver:%s' % 'SeisCloud',
             ]

        return GreensTensor(traces=[trace for trace in stream],
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)
