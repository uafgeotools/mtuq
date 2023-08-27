
import numpy as np

from matplotlib import pyplot
from obspy.geodetics import gps2dist_azimuth


def station_label_writer(ax, station, origin, units='km'):
    """ Displays station id, distance, and azimuth to the left of current axes
    """
    distance_in_m, azimuth, _ = gps2dist_azimuth(
        origin.latitude,
        origin.longitude,
        station.latitude,
        station.longitude)

    # construct station label
    if len(station.location) > 1:
        label = station.network + '.' + station.station + '.' + station.location
    else:
        label = station.network + '.' + station.station

    # display station label
    pyplot.text(0.2,0.50, label, fontsize=11, transform=ax.transAxes)

    # display distance
    if units=='m':
        label = '%d m' % round(distance_in_m)

    elif units=='km':
        label = '%d km' % round(distance_in_m/1000.)

    elif units=='deg':
        label = '%d%s' % (round(m_to_deg(distance_in_m)), u'\N{DEGREE SIGN}')

    pyplot.text(0.2,0.35, label, fontsize=11, transform=ax.transAxes)

    # display azimuth
    azimuth =  '%d%s' % (round(azimuth), u'\N{DEGREE SIGN}')
    pyplot.text(0.2,0.20, azimuth, fontsize=11, transform=ax.transAxes)


def trace_label_writer(axis, dat, syn, total_misfit=1.):
    """ Displays cross-correlation time shift and misfit information below trace
    """
    ymin = axis.get_ylim()[0]

    s = syn.data
    d = dat.data

    # display cross-correlation time shift
    time_shift = 0.
    time_shift += _getattr(syn, 'time_shift', np.nan)
    time_shift += _getattr(dat, 'static_time_shift', 0)
    axis.text(0.,(1/4.)*ymin, '%.2f' %time_shift, fontsize=11)

    # display maximum cross-correlation coefficient
    Ns = np.dot(s,s)**0.5
    Nd = np.dot(d,d)**0.5
    if Ns*Nd > 0.:
        max_cc = np.correlate(s, d, 'valid').max()
        max_cc /= (Ns*Nd)
        axis.text(0.,(2/4.)*ymin, '%.2f' %max_cc, fontsize=11)
    else:
        max_cc = np.nan
        axis.text(0.,(2/4.)*ymin, '%.2f' %max_cc, fontsize=11)

    # display percent of total misfit
    misfit = _getattr(syn, 'misfit', np.nan)
    misfit /= total_misfit
    if misfit >= 0.1:
        axis.text(0.,(3/4.)*ymin, '%.1f' %(100.*misfit), fontsize=11)
    else:
        axis.text(0.,(3/4.)*ymin, '%.2f' %(100.*misfit), fontsize=11)


#
# utility functions
#

def _getattr(trace, name, *args):
    if len(args)==1:
        if not hasattr(trace, 'attrs'):
            return args[0]
        else:
            return getattr(trace.attrs, name, args[0])
    elif len(args)==0:
        return getattr(trace.attrs, name)
    else:
        raise TypeError("Wrong number of arguments")


