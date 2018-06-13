

from obspy.geodetics import gps2dist_azimuth, kilometers2degrees



def distance(station, origin):
    if hasattr(station, 'latitude') and\
        hasattr(station, 'longitude'):
        distance_in_m, _, _, = gps2dist_azimuth(
            station.latitude,
            station.longitude,
            origin.latitude,
            origin.longitude)
        return distance_in_m/1000.

    else:
        raise Exception


def distance_azimuth(station, origin):
    if hasattr(station, 'latitude') and\
        hasattr(station, 'longitude'):
        distance_in_m, azimuth, _, = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)
        return distance_in_m/1000., azimuth

    else:
        raise Exception


def km2deg(distance_in_km):
    return kilometers2degrees(distance_in_km, radius=6371.)
