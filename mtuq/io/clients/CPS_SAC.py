
import obspy
import numpy as np

from os.path import basename, exists, isdir, join
from os import listdir
from mtuq.greens_tensor.FK import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth


# FIX FOR CPS
# An FK simulation outputs 12 SAC files each with filename extensions
# 0,1,2,3,4,5,6,7,8,9,a,b.  The SAC files ending in .2 and .9 contain
# only zero data, so we exclude them from the following list.
# The order of traces in the list is the order in which CAP stores
# the time series.

# A CPS run where the file96 results are converted to SAC files produces

POSITION = [
    '8',  '5',             # t
    '10', '7', '4', '2',   # r
    '9',  '6', '3', '1',   # z
]

CHANNELS = [
    'TSS', 'TDS',
    'REX', 'RSS', 'RDS', 'RDD',
    'ZEX', 'ZSS', 'ZDS', 'ZDD',
]


class Client(ClientBase):
    """  CPS database client

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.CPS_SAC import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

      `GreensTensor`s are obtained by reading precomputed time series from an 
      CPS directory tree.  



      Such trees contain SAC files organized by model, 
      event depth, and event distance, as used by the `Zhu1994`
      software packages.

    """

    def __init__(self, path_or_url=None, model=None,
                 include_mt=True, include_force=False):

        if not path_or_url:
            raise Exception

        if not exists(path_or_url):
            raise Exception

        if include_force:
            raise NotImplementedError

        if not model:
            model = basename(path_or_url)

        # path to CPS directory tree
        self.path = path_or_url

        # model from which CPS Green's functions were computed
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force

    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Extracts Green's tensors from database

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

        # dep = str(int(round(origin.depth_in_m/1000.)))
        # dep = str(int(np.ceil(origin.depth_in_m/1000.))).zfill(4)
        dep_desired = "{:06.1f}".format(
            np.ceil(origin.depth_in_m/1000.) * 10)[:-2]
        # dep_folder_desired = dep[:-2]

        # dst = str(int(round(distance_in_m/1000.)))
        # dst = str(int(np.ceil(distance_in_m/1000.))).zfill(4)
        dst_desired = "{:07.1f}".format(np.ceil(distance_in_m/1000.) * 10)[:-2]

        if self.include_mt:

            # Review all folders in CPS Green's Function directory. Folder
            # names correspond with depth of source. Find the folder
            # with a value closest to the one we are after.
            all_entries = listdir(self.path)

            # Filter out folder names that are numeric
            numeric_folder_names = [entry for entry in all_entries
                                    if entry.isdigit() and isdir(join(self.path, entry))]

            # Convert numeric folder names to integers
            numeric_folder_names_int = [int(folder)
                                        for folder in numeric_folder_names]
            # Find depth closest to our desired value
            dep_folder = numeric_folder_names[numeric_folder_names_int.index(min(numeric_folder_names_int,
                                                                                 key=lambda x: abs(x - int(dep_desired))))]

            # Find distance closest to our desired value

            # self.path += '/' + dep_folder
            all_files = listdir(self.path + '/' + dep_folder)
            filenames_without_extensions_inline = [
                filename.split('.')[0] for filename in all_files]
            filenames_without_letters = [filename for filename in filenames_without_extensions_inline if not any(
                char.isalpha() for char in filename)]
            filenames_unique = [entry[:5]
                                for entry in list(set(filenames_without_letters))]
            filenames_unique_int = [int(filename)
                                    for filename in filenames_unique]
            dst_value = filenames_unique[filenames_unique_int.index(
                min(filenames_unique_int, key=lambda x: abs(x - int(dst_desired))))]

            for _i, ext in enumerate(CHANNELS):
                trace = obspy.read('%s/%s/%s/%s%s.%s' %
                                   (self.path, self.model,
                                    dep_folder, dst_value, dep_folder, ext),
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
            'solver:%s' % 'CPS',
        ]

        return GreensTensor(traces=[trace for trace in traces],
                            station=station, origin=origin, tags=tags,
                            include_mt=self.include_mt, include_force=self.include_force)
