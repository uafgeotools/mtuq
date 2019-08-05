
import numpy as np
from mtuq.util.math import isclose
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from scipy.signal import fftconvolve


def cut(trace, t1, t2):
    """ 
    trace: obspy trace
    t1: desired start time
    t2: desired end time
    """
    if t1 < float(trace.stats.starttime):
        raise Exception('The chosen window begins before the trace.  Consider '
           'using a later window, or to automatically pad the beginning of the '
           'trace with zeros, use mtuq.util.signal.resample instead')

    if t2 > float(trace.stats.endtime):
        raise Exception('The chosen window ends after the trace.  Consider '
           'using an earlier window, or to automatically pad the end of the '
           'trace with zeros, use mtuq.util.signal.resample instead')

    t0 = float(trace.stats.starttime)
    dt = float(trace.stats.delta)
    it1 = int((t1-t0)/dt)
    it2 = int((t2-t0)/dt)
    trace.data = trace.data[it1:it2]
    trace.stats.starttime = t1
    trace.stats.npts = it2-it1


def resample(data, t1_old, t2_old, dt_old, t1_new, t2_new, dt_new):
    """ 
    data: numpy array
    t1_new: desired start time for resampled data
    t2_new: desired end time for resampled data
    dt_new: desired time increment for resampled data
    """
    dt = dt_old

    tmp1 = round(t1_old/dt)*dt
    tmp2 = round((t2_old-t1_old)/dt)*dt
    t1_old = tmp1
    t2_old = tmp1 + tmp2

    tmp1 = round(t1_new/dt)*dt
    tmp2 = round((t2_new-t1_new)/dt)*dt
    t1_new = tmp1
    t2_new = tmp1 + tmp2

    # offset between new and old start times
    i1 = int(round((t1_old-t1_new)/dt))

    # offset between new and old end times
    i2 = int(round((t2_old-t2_new)/dt))

    nt = int(round((t2_new-t1_new)/dt))
    resampled_data = np.zeros(nt+1)

    # cut both ends
    if t1_old <= t1_new <= t2_new <= t2_old:
        resampled_data[0:nt] = data[i1:i1+nt]

    # cut beginning only
    elif t1_old <= t1_new <= t2_old <= t2_new:
        resampled_data[0:nt+i2] = data[i1:]

    # cut end only
    elif t1_new <= t1_old <= t2_new <= t2_old:
        resampled_data[i1:nt] = data[0:-i2-1]

    # cut neither end
    elif t1_new <= t1_old <= t2_old <= t2_new:
        resampled_data[i1:i2] =  data[0:]


    if dt_old==dt_new:
        return resampled_data
    else:
        t_old = np.linspace(t1_new, t1_new+dt_old*(nt+1), nt+1)
        t_new = np.arange(t1_new, t2_new+dt_new, dt_new)
        return np.interp(t_new, t_old, resampled_data)



def check_time_sampling(stream):
    """ Checks if all traces in stream have the same time sampling
    """
    starttime = [float(trace.stats.starttime) for trace in stream]
    starttime0 = [float(stream[0].stats.starttime)]*len(stream)
    if not isclose(starttime, starttime0):
        return False

    delta = [trace.stats.delta for trace in stream]
    delta0 = [stream[0].stats.delta]*len(stream)
    if not isclose(delta, delta0):
        return False

    npts = [trace.stats.npts for trace in stream]
    npts0 = [stream[0].stats.npts]*len(stream)
    if npts != npts0:
        return False

    return True



def get_arrival(arrivals, phase):
    phases = []
    for arrival in arrivals:
        phases += [arrival.phase.name]

    if phase not in phases:
        raise Exception("Phase not found")

    arrival = arrivals[phases.index(phase)]
    return arrival.time



def get_components(stream):
    components = []
    for trace in stream:
        components += [trace.stats.channel[-1].upper()]
    return components


def get_distance_in_m(station, origin):
    distance_in_m, _, _ = gps2dist_azimuth(
        origin.latitude,
        origin.longitude,
        station.latitude,
        station.longitude)
    return distance_in_m


def get_distance_in_deg(station, origin):
    return m_to_deg(
        get_distance_in_m(station, origin))


def get_time_sampling(stream):
    if len(stream) > 0:
        npts = stream[0].data.size
        dt = stream[0].stats.delta
    else:
        npts = None
        dt = None
    return npts, dt


def m_to_deg(distance_in_m):
    return kilometers2degrees(distance_in_m/1000., radius=6371.)



#
# optimized cross-correlation functions
#

def correlate(v1, v2):
    # fast cross-correlation of unpadded array v1 and padded array v2
    n1, n2 = len(v1), len(v2)

    if n1>2000 or n2-n1>200:
        # for long traces, frequency-domain implementation is usually faster
        return fftconvolve(v1, v2[::-1], 'valid')
    else:
        # for short traces, time-domain implementation is usually faster
        return np.correlate(v1, v2, 'valid')



def corr_nd1_nd2(data, greens, time_shift_max):
    # correlates 1D and 2D data structures
    corr_all = []

    for d, g in zip(data, greens):

        ncomp = len(g.components)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[1]

        # array that will hold correlations
        corr = np.zeros((ncomp, ngf, 2*npts_padding+1))

        # the main work starts now
        for _i, component in enumerate(g.components):
            trace = d.select(component=component)[0]

            for _j in range(ngf):
                corr[_i, _j, :] =\
                    correlate(array[_i, _j, :], trace.data)

        corr_all += [corr]

    return corr_all


def autocorr_nd1(data, time_shift_max):
    # autocorrelates 1D data strucutres (reduces to dot product)
    corr_all = []

    for d in data:
        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        corr = np.zeros(ncomp)

        # the main work starts now
        for _i1, trace in enumerate(d):
            corr[_i1] = np.dot(trace.data, trace.data)

        corr_all += [corr]

    return corr_all


def autocorr_nd2(greens, time_shift_max):
    # autocorrelates 2D data structures
    corr_all = []

    for g in greens:
        ncomp = len(g.components)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = get_time_sampling(g)
        npts_padding = int(time_shift_max/dt)

        ones = np.pad(np.ones(npts-2*npts_padding), 2*npts_padding, 'constant')

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[1]

        # array that will hold correlations
        corr = np.zeros((ncomp, 2*npts_padding+1, ngf, ngf))

        # the main work starts now
        for _i in range(ncomp):
            for _j1 in range(ngf):
                for _j2 in range(ngf):

                    if _j1<=_j2:
                        # calculate upper elements
                        corr[_i, :, _j1, _j2] = correlate(
                            array[_i, _j1, :]*array[_i, _j2, :], ones)

                    else:
                        # fill in lower elements by symmetry
                        corr[_i, :, _j1, _j2] = corr[_i, :, _j2, _j1]

        corr_all += [corr]

    return corr_all


def corr_init(data, time_shift_max):
    # allocates arrays to hold correlations
    corr_all = []

    for d in data:
        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        corr_all += [np.zeros(2*npts_padding+1)]

    return corr_all



