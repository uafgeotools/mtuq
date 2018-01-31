
class Factory(object):
    """ 
    Reads precomputed Green's tensors from a netCDF file produced by AxiSEM.
    The resulting Green's tensors are stored in an mtuq GreensTensorList, 
    similar to how traces are stored in an obspy Stream.

    Reading Green's tensors is a two-step procedure:
      1) greens_tensor_factory = mtuq.greens.instaseis.Factory(path, buffer_type)
      2) greens_tensor_list = greens_tensor_factory(stations, origins) 

    In the first step, one supplies the path to the AxiSEM netCDF file and a
    buffer flag that specifies whether to read Green's tensors fully into
    memory or rely on Instaseis buffering.

    In the second step, one supplies a list of stations and event origins.
    A GreensTensor object will be created for each station-event pair.

    :input path: path to Green's function directory tree
    :type path: str
    :input model: name of Earth model
    :type model: str
    """
    def __init__(self, path, buffer_type='full', buffer_size=0):
        if not exists(path):
            raise Exception

        self.buffer_type = buffer_type

        # instatiate database
        self.db = ForwardInstaseisDB(db_path, buffer_size=buffer_size)


    def __call__(self, stations, origins):
        """
        Reads Green's tensors corresponding to given stations and origins
        """
        greens_tensor_list = mtuq.greens.base.GreensTensorList()
        station_ids = ['']

        for origin in origins:
            for stats in stations:
                station_id = stats.network+'.'+stats.station
                if station_id != stations_ids[-1]:
                    if self.buffer_type=='full':
                        greens_tensor_list += _read_greens_tensor(
                            stats, origin)
                    else:
                        greens_tensor_list += GreensTensorPartlyBuffered(
                            [], stats, origin)
                    station_ids += [station_id]
                else:
                    # if the station_id is the same, no need to reread;
                    # typically this means there is a station with
                    # multiple components
                    data = greens_tensor_list[-1].data
                    greens_tensor_list += GreensTensor(data, stats)

        return greens_tensor_list




class GreensTensorFullyBuffered(object):
    """ Reads Green's tensors fully into memory
    """
    def combine(self, mt):
        raise NotImplementedError


class GreensTensorPartlyBuffered(object):
    """ Relies on Instaseis buffering capabilities to reduce I/O
    """
    def __init__(self, data, stats, origin):
        " instantiates instaseis db object"
        self.db = data
        self.stats = stats

    def combine(self, mt):
        return self.db.get_seismogram(
            _receiver(self.stats), 
            _source(mt, self.origin),
            _component(self.stats))


def _generate_greens_tensor(stats, origin):
    raise NotImplementedError



def _receiver(stats)
    "obspy.core.Stats --> instaseis.Receiver"
    raise NotImplementedError


def _source(mt, origin)
    "obspy.core.source.Origin --> instaseis.Source"
    raise NotImplementedError


