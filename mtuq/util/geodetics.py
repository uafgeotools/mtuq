

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
