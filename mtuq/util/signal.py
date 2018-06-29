
import numpy as np
from mtuq.util.math import isclose

def convolve(data, wavelet, overwrite=True):
    """
    data: obspy stream
    wavelet: numpy array
    """
    if overwrite:
        convolved_data = data
    else:
        convolved_data = deepcopy(data)

    for _i in range(len(data)):
        convolved_data[_i].data = np.convolve(data[_i].data, wavelet, mode='same')

    return convolved_data


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
    if dt_old != dt_new:
        raise NotImplementedError
    dt = dt_new

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

    #FIXME: NEED TO DOUBLE CHECK ALL CASES
    # cut both ends
    if t1_old <= t1_new <= t2_new <= t2_old:
        resampled_data[0:nt] = data[i1:i2]

    # cut beginning only
    elif t1_old <= t1_new <= t2_old <= t2_new:
        resampled_data[0:nt+i2] = data[i1:]

    # cut end only
    elif t1_new <= t1_old <= t2_new <= t2_old:
        resampled_data[i1:nt] = data[0:-i2-1]

    # cut neither end
    elif t1_new <= t1_old <= t2_old <= t2_new:
        resampled_data[i1:i2] =  data[0:]

    return resampled_data


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

